import sqlite3
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

SQLITE_PATH = Path("pes_kariyer.db")

# 1) Buraya Supabase URI'ni yapıştır (Streamlit secrets'teki ile aynı)
DATABASE_URL = "postgresql://postgres.ugpwrtmsblzgyopvtcss:Eoracle1717.@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"

def main():
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(f"{SQLITE_PATH} bulunamadı. migrate dosyasını db ile aynı klasöre koy.")

    src = sqlite3.connect(SQLITE_PATH)
    players = pd.read_sql_query("SELECT player_id, name, pos FROM players", src)
    stats = pd.read_sql_query("""
        SELECT stat_id, player_id, season, scope, mp, goals, assists, rating
        FROM stats
    """, src)
    src.close()

    eng = create_engine(DATABASE_URL, pool_pre_ping=True)

    with eng.begin() as conn:
        # tablolar yoksa oluştur (Postgres)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS players (
                player_id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                pos TEXT NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stats (
                stat_id BIGSERIAL PRIMARY KEY,
                player_id BIGINT NOT NULL REFERENCES players(player_id) ON DELETE CASCADE,
                season TEXT NOT NULL,
                scope TEXT NOT NULL,
                mp INTEGER NOT NULL DEFAULT 0,
                goals INTEGER NOT NULL DEFAULT 0,
                assists INTEGER NOT NULL DEFAULT 0,
                rating DOUBLE PRECISION,
                UNIQUE(player_id, season, scope)
            );
        """))

        # Temizle (aynı datayı iki kere basmayalım)
        conn.execute(text("TRUNCATE TABLE stats RESTART IDENTITY CASCADE;"))
        conn.execute(text("TRUNCATE TABLE players RESTART IDENTITY CASCADE;"))

    # player_id’leri koruyarak basacağız (FK bozulmasın)
    players.to_sql("players", eng, if_exists="append", index=False, method="multi")
    stats.to_sql("stats", eng, if_exists="append", index=False, method="multi")

    print("✅ Migrate bitti: SQLite -> Postgres")

if __name__ == "__main__":
    main()
