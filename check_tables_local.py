import psycopg2

conn = psycopg2.connect('postgresql://postgres:zaima@localhost:5432/imci_ai_prod')
cur = conn.cursor()
cur.execute("""
    SELECT table_schema, table_name 
    FROM information_schema.tables 
    WHERE table_name IN ('location_disease_forecasts', 'weekly_monthly_recommendations', 'location_disease_recommendations')
    ORDER BY table_schema, table_name;
""")
print("Tables on 5432:")
for row in cur.fetchall():
    print(f"  {row[0]}.{row[1]}")

cur.execute("""
    SELECT table_schema, column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name IN ('weekly_monthly_recommendations', 'location_disease_recommendations')
    ORDER BY table_schema, ordinal_position;
""")
print("\nColumns:")
for row in cur.fetchall():
    print(f"  [{row[0]}.{row[1]}] {row[2]}")

conn.close()
