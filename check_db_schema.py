import os
import psycopg2
from dotenv import load_dotenv

load_dotenv("test/.env")

url = os.getenv("DATABASE_URL")
schema = os.getenv("DB_SCHEMA")
table = os.getenv("RECOMMENDATION_TABLE")
forecast_table = os.getenv("DB_TABLE")

print(f"DATABASE_URL: {url}")
print(f"DB_SCHEMA: {schema}")
print(f"DB_TABLE: {forecast_table}")
print(f"RECOMMENDATION_TABLE: {table}")

conn = psycopg2.connect(url)
cur = conn.cursor()

# Find schemas where these tables exist
cur.execute("""
    SELECT table_schema, table_name 
    FROM information_schema.tables 
    WHERE table_name IN ('location_disease_forecasts', 'weekly_monthly_recommendations', 'location_disease_recommendations')
    ORDER BY table_schema, table_name;
""")
print("\nExisting tables in database:")
for row in cur.fetchall():
    print(f"  {row[0]}.{row[1]}")

# Check columns of weekly_monthly_recommendations if it exists anywhere
cur.execute("""
    SELECT table_schema, column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'weekly_monthly_recommendations'
    ORDER BY table_schema, ordinal_position;
""")
print("\nColumns of weekly_monthly_recommendations:")
for row in cur.fetchall():
    print(f"  [{row[0]}] {row[1]} ({row[2]})")

conn.close()
