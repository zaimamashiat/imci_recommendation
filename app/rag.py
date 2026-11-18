import hashlib
import logging
import pickle
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app import config

logger = logging.getLogger("imci_rag")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file to detect modifications."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


class IMCIRAG:
    """
    RAG pipeline for Integrated Management of Childhood Illness (IMCI) Chart Booklet.
    Extracts pages from the PDF, generates overlapping chunks with page citations,
    indexes semantic embeddings using SentenceTransformers, and persists/caches the
    index to disk for fast startup.
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_path: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.pdf_path = Path(pdf_path or config.IMCI_PDF_PATH)
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.cache_path = Path(cache_path or config.INDEX_CACHE_PATH)
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        self._model: Optional[SentenceTransformer] = None
        self._model_lock = threading.Lock()
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}

    @property
    def model(self) -> SentenceTransformer:
        """Thread-safe lazy loader for the SentenceTransformer model with local cache preference."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    logger.info("Loading embedding model: %s", self.model_name)
                    try:
                        # Attempt to load from local cache first to avoid remote network check overhead
                        self._model = SentenceTransformer(self.model_name, local_files_only=True)
                    except Exception:
                        self._model = SentenceTransformer(self.model_name)
        return self._model

    def warm_model(self) -> None:
        """Pre-warm the embedding model into memory."""
        _ = self.model

    # -------------------------------------------------------------------------
    # PDF EXTRACTION & CHUNKING
    # -------------------------------------------------------------------------

    def extract_and_chunk(self) -> List[Dict[str, Any]]:
        """
        Extract text from all PDF pages, identify scanned/empty pages,
        and generate overlapping text chunks with original page references.
        """
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"IMCI PDF not found at: {self.pdf_path.resolve()}")

        logger.info("Extracting text from PDF: %s", self.pdf_path.name)
        reader = PdfReader(str(self.pdf_path))
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError(f"The PDF file {self.pdf_path.name} is empty (0 pages).")

        extracted_chunks: List[Dict[str, Any]] = []
        readable_pages = set()
        empty_pages = []

        chunk_id = 0
        step = max(100, self.chunk_size - self.chunk_overlap)

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            clean_text = re.sub(r"\s+", " ", raw_text).strip()

            if not clean_text or len(clean_text) < 20:
                empty_pages.append(page_num)
                continue

            readable_pages.add(page_num)

            for start in range(0, len(clean_text), step):
                chunk_text = clean_text[start : start + self.chunk_size].strip()
                if not chunk_text:
                    continue

                chunk_id += 1
                extracted_chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page_num,
                    "page_ref": f"Page {page_num}",
                    "text": chunk_text,
                })

        logger.info(
            "PDF processed: %d total pages, %d readable pages with text, %d chunks created.",
            total_pages,
            len(readable_pages),
            len(extracted_chunks),
        )

        if empty_pages:
            logger.warning(
                "%d pages had no extractable text (potential scanned pages): %s",
                len(empty_pages),
                empty_pages[:10],
            )

        if not extracted_chunks:
            raise ValueError(
                f"No selectable text could be extracted from {self.pdf_path.name}. "
                "The PDF might contain scanned images requiring OCR preprocessing."
            )

        self.metadata = {
            "total_pages": total_pages,
            "readable_pages": len(readable_pages),
            "empty_pages_count": len(empty_pages),
            "total_chunks": len(extracted_chunks),
        }
        return extracted_chunks

    # -------------------------------------------------------------------------
    # VECTOR INDEXING & CACHING
    # -------------------------------------------------------------------------

    def build_index(self, force_rebuild: bool = False, prewarm_model: bool = True) -> None:
        """
        Build or load the vector index.
        Uses cached embeddings if the PDF checksum and configuration match.
        """
        if self.vectors is not None and not force_rebuild:
            logger.info("Vector index already loaded in memory.")
            if prewarm_model:
                self.warm_model()
            return

        current_hash = compute_file_sha256(self.pdf_path)

        # 1. Try loading cached index from disk
        if not force_rebuild and self.cache_path.is_file():
            try:
                with open(self.cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                if (
                    cache_data.get("pdf_hash") == current_hash
                    and cache_data.get("model_name") == self.model_name
                    and cache_data.get("chunk_size") == self.chunk_size
                    and cache_data.get("chunk_overlap") == self.chunk_overlap
                    and "vectors" in cache_data
                    and "chunks" in cache_data
                ):
                    self.vectors = cache_data["vectors"]
                    self.chunks = cache_data["chunks"]
                    self.metadata = cache_data.get("metadata", {})
                    logger.info(
                        "Loaded cached RAG index from %s (%d chunks, %d dims).",
                        self.cache_path,
                        len(self.chunks),
                        self.vectors.shape[1],
                    )
                    if prewarm_model:
                        self.warm_model()
                    return
                else:
                    logger.info("Cached index signature mismatch or PDF modified. Rebuilding...")
            except Exception as exc:
                logger.warning("Failed to load cached index (%s). Rebuilding from PDF...", exc)

        # 2. Build index from scratch
        logger.info("Extracting and building new index for: %s", self.pdf_path.name)
        self.chunks = self.extract_and_chunk()

        texts_to_embed = [
            f"[IMCI Booklet Page {c['page_number']}]\n{c['text']}"
            for c in self.chunks
        ]

        logger.info("Computing embeddings for %d text chunks...", len(texts_to_embed))
        embeddings = self.model.encode(
            texts_to_embed,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.vectors = np.asarray(embeddings, dtype=np.float32)

        # 3. Persist index to cache
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {
                        "pdf_hash": current_hash,
                        "model_name": self.model_name,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "chunks": self.chunks,
                        "vectors": self.vectors,
                        "metadata": self.metadata,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info("Successfully persisted RAG index to %s", self.cache_path)
        except Exception as exc:
            logger.warning("Could not persist index cache to %s: %s", self.cache_path, exc)

    def build(self, force_rebuild: bool = False) -> None:
        """Alias for build_index."""
        self.build_index(force_rebuild=force_rebuild)

    # -------------------------------------------------------------------------
    # SEMANTIC RETRIEVAL
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant passages for an arbitrary semantic query.
        """
        if self.vectors is None:
            self.build_index()

        clean_query = query.strip()
        if not clean_query:
            return []

        query_vector = self.model.encode(
            [clean_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        query_vector = np.asarray(query_vector, dtype=np.float32)

        scores = np.dot(self.vectors, query_vector)
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        seen_texts = set()

        for idx in ranked_indices:
            score = float(scores[idx])
            if score < min_similarity:
                continue

            chunk = self.chunks[int(idx)]
            preview = chunk["text"][:100]
            if preview in seen_texts:
                continue
            seen_texts.add(preview)

            results.append({
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "page": f"Page {chunk['page_number']}",
                "text": chunk["text"],
                "similarity": round(score, 4),
            })

            if len(results) >= top_k:
                break

        return results

    # -------------------------------------------------------------------------
    # DISEASE-SPECIFIC RETRIEVAL
    # -------------------------------------------------------------------------

    def retrieve_for_disease(
        self,
        disease_name: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant IMCI booklet passages tailored for government/public-health planning.

        Strategy:
        1. Query with disease-specific public-health terms (prevention, nutrition, surveillance,
           community interventions, service delivery) to bias away from individual clinical steps.
        2. Apply relevance filtering: if the best passage is strong (>=0.50), drop passages
           more than 0.20 below the top score.
        3. Post-filter: deprioritise passages that read as individual clinical prescriptions
           (dosage tables, drug regimens for a single patient) while keeping passages about
           facility readiness, community counselling, and preventive interventions.
        4. Never return fewer than 1 passage; fall back to a broad disease query.
        """
        top_k = top_k or config.RETRIEVAL_TOP_K

        # Public-health-oriented query terms per disease
        disease_keywords = {
            "anaemia": (
                "anaemia prevention iron folate supplementation nutrition counselling "
                "community screening vitamin A micronutrient palmar pallor health system "
                "referral service delivery caregiver advice feeding practices"
            ),
            "malnutrition": (
                "malnutrition prevention community screening muac mid-upper arm circumference "
                "nutrition programme supplementary feeding caregiver counselling breastfeeding "
                "complementary feeding growth monitoring severe acute malnutrition"
            ),
            "pneumonia": (
                "pneumonia prevention vaccination immunisation chest indrawing fast breathing "
                "amoxicillin referral oxygen health facility community awareness danger signs"
            ),
            "diarrhoea": (
                "diarrhoea prevention oral rehydration ors zinc handwashing water sanitation "
                "community hygiene dehydration signs caregiver counselling community programme"
            ),
            "fever": (
                "fever malaria prevention rapid diagnostic test rdt community screening "
                "antimalarial treatment referral danger signs caregiver counselling"
            ),
            "ear infection": (
                "ear infection prevention community awareness caregiver counselling discharge "
                "referral health facility management"
            ),
            "measles": (
                "measles prevention vaccination immunisation community outbreak surveillance "
                "referral danger signs caregiver counselling vitamin A"
            ),
        }

        d_lower = disease_name.strip().lower()
        extra_terms = ""
        for key, terms in disease_keywords.items():
            if key in d_lower:
                extra_terms = terms
                break

        # Public-health-framed query (more tokens than top-k to allow filtering)
        query = (
            f"IMCI public health interventions for {disease_name}. "
            f"Government prevention programmes, community screening, nutrition counselling, "
            f"health facility service delivery, caregiver education, community awareness. "
            f"{extra_terms}"
        ).strip()

        fetch_k = min(top_k + 4, 10)   # fetch a few extras so we can filter
        logger.info(
            "Retrieving public-health IMCI guidance for disease: %s (top_k=%d, fetch_k=%d)",
            disease_name, top_k, fetch_k,
        )
        results = self.retrieve(query=query, top_k=fetch_k, min_similarity=min_similarity)

        # --- Relevance window filter ---
        if results and results[0]["similarity"] >= 0.40:
            top_score = results[0]["similarity"]
            threshold = max(0.35, top_score - 0.20)
            results = [r for r in results if r["similarity"] >= threshold]

        # --- Deprioritise purely individual-treatment passages ---
        # Passages dominated by drug dosage tables / patient-level prescriptions are
        # moved to the end so the top_k slice prefers prevention/planning content.
        CLINICAL_PRESCRIPTION_SIGNALS = [
            "mg/kg", "per kg", "dose:", "doses:", "tablet dose", "syrup dose",
            "give oral", "give intramuscular", "give rectal", "give iv",
            "first dose", "second dose", "complete treatment",
            "treatment a:", "treatment b:", "treatment c:",
            "plan a:", "plan b:", "plan c:",
        ]
        priority, deferred = [], []
        for r in results:
            text_lower = r["text"].lower()
            is_clinical_rx = sum(1 for sig in CLINICAL_PRESCRIPTION_SIGNALS if sig in text_lower) >= 2
            (deferred if is_clinical_rx else priority).append(r)

        ordered = priority + deferred
        trimmed = ordered[:top_k]

        # Fallback: if nothing survived, do a plain broad search
        if not trimmed:
            logger.info("Fallback broader search for: %s", disease_name)
            trimmed = self.retrieve(query=disease_name, top_k=top_k, min_similarity=0.10)

        logger.info(
            "Final passage selection for '%s': %d passages, pages: %s",
            disease_name,
            len(trimmed),
            [r.get("page") for r in trimmed],
        )
        return trimmed

    def get_context(
        self,
        disease_name: str,
        top_k: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get formatted context text and structured passages for LLM consumption.
        Limits total context to approximately 4,000–6,000 characters while preserving citations.
        """
        max_chars = max_context_chars or config.MAX_CONTEXT_CHARS
        passages = self.retrieve_for_disease(disease_name, top_k=top_k)

        context_parts = []
        current_len = 0
        pages_included = []

        for p in passages:
            page_ref = p.get("page") or f"Page {p['page_number']}"
            block = f"[{page_ref}]\n{p['text']}"
            if current_len + len(block) > max_chars and context_parts:
                break
            context_parts.append(block)
            current_len += len(block)
            pages_included.append(page_ref)

        return {
            "context": "\n\n---\n\n".join(context_parts),
            "passages": passages,
            "pages": sorted(set(pages_included)),
            "total_context_chars": current_len,
        }

    def get_relevant_context(self, disease_name: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Alias for test.py compatibility."""
        return self.get_context(disease_name, top_k=top_k)


# -----------------------------------------------------------------------------
# SHARED THREAD-SAFE SINGLETON RAG INSTANCE
# -----------------------------------------------------------------------------

_shared_rag_instance: Optional[IMCIRAG] = None
_rag_singleton_lock = threading.Lock()


def get_rag_instance(force_rebuild: bool = False) -> IMCIRAG:
    """
    Return a shared in-memory singleton instance of IMCIRAG.
    Guarantees thread-safe single initialization on startup and avoids repeated model loading.
    """
    global _shared_rag_instance
    if _shared_rag_instance is None or force_rebuild:
        with _rag_singleton_lock:
            if _shared_rag_instance is None or force_rebuild:
                rag = IMCIRAG()
                rag.build_index(force_rebuild=force_rebuild, prewarm_model=True)
                _shared_rag_instance = rag
    return _shared_rag_instance


# Backwards compatibility alias
IMCIKnowledgeBase = IMCIRAG
