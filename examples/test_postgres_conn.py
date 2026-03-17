import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print(cur.fetchone())