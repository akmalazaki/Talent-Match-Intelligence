import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import re

# === Load environment variables ===
load_dotenv()

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

excel_path = "Study Case DA.xlsx"

try:
    conn = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        sslmode="require"
    )
    print("✅ Connection successful!")

    cursor = conn.cursor()
    xlsx = pd.ExcelFile(excel_path)
    print(f"📚 Ditemukan sheet: {xlsx.sheet_names}")

    for sheet in xlsx.sheet_names:
        print(f"\n📄 Mengimpor sheet: {sheet}")

        # Bersihkan nama tabel dari karakter aneh
        table_name = re.sub(r'[^0-9a-zA-Z_]+', '_', sheet.strip().lower())
        table_name = re.sub(r'_+', '_', table_name)  # hilangkan underscore ganda

        df = pd.read_excel(excel_path, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Tambahkan kolom id jika belum ada
        if "id" not in df.columns:
            df.insert(0, "id", range(1, len(df) + 1))

        # Buat tabel baru (drop kalau sudah ada)
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')

        # Buat struktur tabel
        create_sql_parts = []
        for col in df.columns:
            dtype = "TEXT"
            if np.issubdtype(df[col].dtype, np.integer):
                dtype = "BIGINT"
            elif np.issubdtype(df[col].dtype, np.floating):
                dtype = "DOUBLE PRECISION"
            elif np.issubdtype(df[col].dtype, np.datetime64):
                dtype = "TIMESTAMP"
            elif df[col].dtype == bool:
                dtype = "BOOLEAN"
            create_sql_parts.append(f'"{col}" {dtype}')

        create_sql = f'CREATE TABLE "{table_name}" ({", ".join(create_sql_parts)}, PRIMARY KEY (id));'
        cursor.execute(create_sql)
        print(f"🛠️ Struktur tabel '{table_name}' dibuat.")

        # Insert data
        for _, row in df.iterrows():
            cols = ", ".join([f'"{c}"' for c in df.columns])
            vals = ", ".join(["%s"] * len(df.columns))
            insert_sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({vals})'
            cursor.execute(insert_sql, tuple(row))
        conn.commit()

        print(f"✅ Sheet '{sheet}' berhasil diunggah ke tabel '{table_name}' ({len(df)} baris).")

    cursor.close()
    conn.close()
    print("\n🎉 Semua sheet berhasil diimpor ke Supabase!")

except Exception as e:
    print(f"❌ Failed to connect or import: {e}")
