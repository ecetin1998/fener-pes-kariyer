import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_searchbox import st_searchbox

DB_PATH = Path("pes_kariyer.db")
LOGO_PATH = Path("fener_logo.png")  # aynı klasör

SEASONS = [f"{y}-{str((y+1) % 100).zfill(2)}" for y in range(2025, 2040)]
POSITIONS = ["kl", "stp", "slb", "sğb", "dos", "gö", "oos", "sla", "sğa", "sf"]
TOURNAMENTS = ["Lig", "Kupa", "Avrupa"]
SCOPES = ["Genel", "Lig", "Kupa", "Avrupa"]

LINEUP = ["kl", "stp", "stp", "slb", "sğb", "dos", "gö", "oos", "sla", "sğa", "sf"]

PAGES = {
    "Giriş": "0) Giriş",
    "Yeni Oyuncu": "1) Yeni oyuncu ekle",
    "Mevcut Oyuncu": "2) Mevcut oyuncuya giriş",
    "İstatistikler": "3) İstatistikler",
    "Oyuncu Form": "4) Oyuncu Form",
    "Rekorlar": "5) Rekorlar",
}

# ---------------- Colors / Theme ----------------
FENER_BG = "#0b1f3a"
FENER_PANEL = "#102a4d"
FENER_YELLOW = "#ffd400"
FENER_BORDER = "#ffd400"
FENER_GREEN_PITCH = "#0f6b3e"
FENER_GREEN_LINE = "#bde5c8"


# ---------------- DB (HYBRID: SQLite local, Postgres cloud) ----------------
from sqlalchemy import create_engine, text

def using_postgres() -> bool:
    try:
        return bool(st.secrets.get("postgresql://postgres.ugpwrtmsblzgyopvtcss:Eoracle1717.@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"))
    except Exception:
        return False

_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(st.secrets["postgresql://postgres.ugpwrtmsblzgyopvtcss:Eoracle1717.@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"], pool_pre_ping=True)
    return _ENGINE

# --- SQLite helpers (eski düzen) ---
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# --- Postgres exec/read ---
def pg_exec(sql: str, params: dict | None = None):
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), params or {})

def pg_read(sql: str, params: dict | None = None) -> pd.DataFrame:
    eng = get_engine()
    return pd.read_sql_query(text(sql), eng, params=params or {})

# ---------------- init_db ----------------
def init_db():
    if using_postgres():
        pg_exec("""
        CREATE TABLE IF NOT EXISTS players (
            player_id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            pos TEXT NOT NULL
        );
        """)
        pg_exec("""
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
        """)
    else:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            pos TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stats (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            season TEXT NOT NULL,
            scope TEXT NOT NULL,
            mp INTEGER NOT NULL DEFAULT 0,
            goals INTEGER NOT NULL DEFAULT 0,
            assists INTEGER NOT NULL DEFAULT 0,
            rating REAL,
            UNIQUE(player_id, season, scope),
            FOREIGN KEY(player_id) REFERENCES players(player_id) ON DELETE CASCADE
        );
        """)
        conn.commit()
        conn.close()

# ---------------- players ----------------
def upsert_player(name: str, pos: str):
    name = name.strip()
    if using_postgres():
        pg_exec("""
            INSERT INTO players(name,pos)
            VALUES(:name,:pos)
            ON CONFLICT(name) DO UPDATE SET pos=EXCLUDED.pos
        """, {"name": name, "pos": pos})
    else:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO players(name,pos) VALUES(?,?)", (name, pos))
        cur.execute("UPDATE players SET pos=? WHERE name=?", (pos, name))
        conn.commit()
        conn.close()

def get_players_df():
    if using_postgres():
        return pg_read("SELECT player_id, name, pos FROM players ORDER BY name")
    else:
        conn = get_conn()
        df = pd.read_sql_query("SELECT player_id, name, pos FROM players ORDER BY name COLLATE NOCASE", conn)
        conn.close()
        return df

# ---------------- stats ----------------
def upsert_stat(player_id: int, season: str, scope: str, mp: int, goals: int, assists: int, rating):
    if using_postgres():
        pg_exec("""
        INSERT INTO stats(player_id, season, scope, mp, goals, assists, rating)
        VALUES(:pid,:season,:scope,:mp,:goals,:assists,:rating)
        ON CONFLICT(player_id, season, scope) DO UPDATE SET
            mp=EXCLUDED.mp,
            goals=EXCLUDED.goals,
            assists=EXCLUDED.assists,
            rating=EXCLUDED.rating
        """, {"pid": player_id, "season": season, "scope": scope, "mp": mp, "goals": goals, "assists": assists, "rating": rating})
    else:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO stats(player_id, season, scope, mp, goals, assists, rating)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(player_id, season, scope) DO UPDATE SET
            mp=excluded.mp,
            goals=excluded.goals,
            assists=excluded.assists,
            rating=excluded.rating
        """, (player_id, season, scope, mp, goals, assists, rating))
        conn.commit()
        conn.close()

def delete_player_season(player_id: int, season: str):
    if using_postgres():
        pg_exec("DELETE FROM stats WHERE player_id=:pid AND season=:season", {"pid": player_id, "season": season})
    else:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM stats WHERE player_id=? AND season=?", (player_id, season))
        conn.commit()
        conn.close()

def fetch_one_stat(player_id: int, season: str, scope: str):
    if using_postgres():
        df = pg_read("""
            SELECT mp, goals, assists, rating
            FROM stats
            WHERE player_id=:pid AND season=:season AND scope=:scope
            LIMIT 1
        """, {"pid": player_id, "season": season, "scope": scope})
    else:
        conn = get_conn()
        df = pd.read_sql_query("""
            SELECT mp, goals, assists, rating
            FROM stats
            WHERE player_id=? AND season=? AND scope=?
            LIMIT 1
        """, conn, params=[player_id, season, scope])
        conn.close()
    return None if df.empty else df.iloc[0].to_dict()

def fetch_stats(season: str | None, scope: str):
    if using_postgres():
        q = """
        SELECT p.name, p.pos, s.season, s.scope, s.mp, s.goals, s.assists, s.rating
        FROM stats s
        JOIN players p ON p.player_id = s.player_id
        WHERE s.scope = :scope
        """
        params = {"scope": scope}
        if season and season != "Hepsi":
            q += " AND s.season = :season"
            params["season"] = season
        q += " ORDER BY p.pos, p.name"
        return pg_read(q, params)
    else:
        conn = get_conn()
        params = [scope]
        q = """
        SELECT p.name, p.pos, s.season, s.scope, s.mp, s.goals, s.assists, s.rating
        FROM stats s
        JOIN players p ON p.player_id = s.player_id
        WHERE s.scope = ?
        """
        if season and season != "Hepsi":
            q += " AND s.season = ?"
            params.append(season)
        q += " ORDER BY p.pos, p.name COLLATE NOCASE"
        df = pd.read_sql_query(q, conn, params=params)
        conn.close()
        return df

def fetch_stats_agg(scope: str):
    q_pg = """
    SELECT
        p.name,
        p.pos,
        SUM(s.mp) AS mp,
        SUM(s.goals) AS goals,
        SUM(s.assists) AS assists,
        CASE
            WHEN SUM(CASE WHEN s.rating IS NOT NULL THEN s.mp ELSE 0 END) > 0
            THEN
                SUM(CASE WHEN s.rating IS NOT NULL THEN s.rating * s.mp ELSE 0 END)
                / SUM(CASE WHEN s.rating IS NOT NULL THEN s.mp ELSE 0 END)
            ELSE NULL
        END AS rating
    FROM stats s
    JOIN players p ON p.player_id = s.player_id
    WHERE s.scope = :scope
    GROUP BY p.player_id, p.name, p.pos
    ORDER BY p.pos, p.name
    """
    if using_postgres():
        return pg_read(q_pg, {"scope": scope})
    else:
        conn = get_conn()
        q_sqlite = q_pg.replace(":scope", "?") + " COLLATE NOCASE"
        # sqlite tarafında ORDER BY satırını zaten seninki gibi bırakalım:
        q_sqlite = """
        SELECT
            p.name,
            p.pos,
            SUM(s.mp) AS mp,
            SUM(s.goals) AS goals,
            SUM(s.assists) AS assists,
            CASE
                WHEN SUM(CASE WHEN s.rating IS NOT NULL THEN s.mp ELSE 0 END) > 0
                THEN
                    SUM(CASE WHEN s.rating IS NOT NULL THEN s.rating * s.mp ELSE 0 END)
                    / SUM(CASE WHEN s.rating IS NOT NULL THEN s.mp ELSE 0 END)
                ELSE NULL
            END AS rating
        FROM stats s
        JOIN players p ON p.player_id = s.player_id
        WHERE s.scope = ?
        GROUP BY p.player_id, p.name, p.pos
        ORDER BY p.pos, p.name COLLATE NOCASE
        """
        df = pd.read_sql_query(q_sqlite, conn, params=[scope])
        conn.close()
        return df

def recompute_general_for_player_season(player_id: int, season: str):
    if using_postgres():
        df = pg_read("""
            SELECT mp, goals, assists, rating
            FROM stats
            WHERE player_id=:pid AND season=:season AND scope IN ('Lig','Kupa','Avrupa')
        """, {"pid": player_id, "season": season})
        rows = df.to_records(index=False)
    else:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT mp, goals, assists, rating
            FROM stats
            WHERE player_id=? AND season=? AND scope IN ('Lig','Kupa','Avrupa')
        """, (player_id, season))
        rows = cur.fetchall()
        conn.close()

    if not rows:
        return

    mp_sum = sum(r[0] or 0 for r in rows)
    g_sum  = sum(r[1] or 0 for r in rows)
    a_sum  = sum(r[2] or 0 for r in rows)

    num = 0.0
    den = 0
    for mp, _, _, rt in rows:
        mp = mp or 0
        if rt is None:
            continue
        num += float(rt) * mp
        den += mp
    general_rating = (num / den) if den > 0 else None

    upsert_stat(player_id, season, "Genel", int(mp_sum), int(g_sum), int(a_sum), general_rating)

def fetch_player_scope_season_rows(player_id: int, scope: str):
    if using_postgres():
        return pg_read("""
            SELECT season, mp, goals, assists, rating
            FROM stats
            WHERE player_id=:pid AND scope=:scope
            ORDER BY season
        """, {"pid": player_id, "scope": scope})
    else:
        conn = get_conn()
        df = pd.read_sql_query("""
            SELECT season, mp, goals, assists, rating
            FROM stats
            WHERE player_id=? AND scope=?
            ORDER BY season
        """, conn, params=[player_id, scope])
        conn.close()
        return df

def fetch_season_records(scope: str):
    if using_postgres():
        return pg_read("""
            SELECT p.name, p.pos, s.season, s.mp, s.goals, s.assists, s.rating
            FROM stats s
            JOIN players p ON p.player_id = s.player_id
            WHERE s.scope = :scope
            ORDER BY s.season
        """, {"scope": scope})
    else:
        conn = get_conn()
        df = pd.read_sql_query("""
            SELECT p.name, p.pos, s.season, s.mp, s.goals, s.assists, s.rating
            FROM stats s
            JOIN players p ON p.player_id = s.player_id
            WHERE s.scope = ?
            ORDER BY s.season
        """, conn, params=[scope])
        conn.close()
        return df


# ---------------- Helpers ----------------
def round_cols(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    d = df.copy()
    for c in d.columns:
        cl = str(c).lower()
        if "rating" in cl:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
        if "katkı" in cl or "katki" in cl:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(0)
    return d


def with_rank(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d.insert(0, "#", range(1, len(d) + 1))
    return d


def show_df(df: pd.DataFrame):
    d = round_cols(df)
    d = with_rank(d)
    st.dataframe(d, use_container_width=True, hide_index=True)


def compute_katki(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rating_num"] = pd.to_numeric(out.get("rating"), errors="coerce")
    out["mp_num"] = pd.to_numeric(out.get("mp"), errors="coerce").fillna(0)
    out["g_num"] = pd.to_numeric(out.get("goals"), errors="coerce").fillna(0)
    out["a_num"] = pd.to_numeric(out.get("assists"), errors="coerce").fillna(0)

    out["gol_asist"] = out["g_num"] + out["a_num"]
    out["katki"] = (out["rating_num"].fillna(0) * out["mp_num"] * 10) + (out["g_num"] * 40) + (out["a_num"] * 25)
    out = out.drop(columns=["rating_num", "mp_num", "g_num", "a_num"])
    return out


def add_trend_cols(df_season: pd.DataFrame) -> pd.DataFrame:
    d = df_season.copy()
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    for col in ["mp", "goals", "assists"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
    d["gol_asist"] = d["goals"] + d["assists"]
    d["ΔGol+Asist"] = d["gol_asist"].diff().fillna(0).astype(int)
    d["ΔMP"] = d["mp"].diff().fillna(0).astype(int)
    d["ΔGol"] = d["goals"].diff().fillna(0).astype(int)
    d["ΔAsist"] = d["assists"].diff().fillna(0).astype(int)
    d["ΔRating"] = d["rating"].diff()
    return d


def form_label_from_last3(df_season: pd.DataFrame):
    d = df_season.copy()
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    d = d.dropna(subset=["rating"])
    if len(d) < 2:
        return "Veri az", None
    last = float(d.iloc[-1]["rating"])
    prev_avg = float(d.iloc[-3:-1]["rating"].mean()) if len(d) >= 3 else float(d.iloc[:-1]["rating"].mean())
    diff = last - prev_avg
    if diff >= 0.2:
        return "Yükselişte", diff
    if diff <= -0.2:
        return "Düşüşte", diff
    return "Stabil", diff


def apply_min_mp(df: pd.DataFrame, min_mp: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mp = pd.to_numeric(df.get("mp"), errors="coerce").fillna(0)
    return df[mp >= min_mp].copy()


def avg_mp_of_df(df: pd.DataFrame) -> float:
    if df is None or df.empty or "mp" not in df.columns:
        return 0.0
    s = pd.to_numeric(df["mp"], errors="coerce").fillna(0)
    s = s[s > 0]
    return float(s.mean()) if len(s) else 0.0


def clear_searchbox_state(prefix: str):
    for k in list(st.session_state.keys()):
        if k.startswith(prefix) or k.startswith(f"q_{prefix}") or prefix in k:
            del st.session_state[k]


def pick_player_searchbox(players_df: pd.DataFrame, label: str, key: str):
    players = players_df[["player_id", "name", "pos"]].copy()

    def search_fn(searchterm: str):
        if not searchterm:
            return []
        q = searchterm.strip().lower()
        m = players["name"].str.lower().str.contains(q, na=False)
        res = players[m].head(12)
        return [f"{r['name']} ({r['pos']})|{int(r['player_id'])}" for _, r in res.iterrows()]

    picked = st_searchbox(
        search_fn,
        key=key,
        label=label,
        placeholder="İsim yaz...",
        clear_on_submit=True,
    )

    if not picked:
        return None, None

    try:
        pid = int(str(picked).split("|")[-1])
        row = players_df[players_df["player_id"] == pid].iloc[0]
        return row["name"], int(pid)
    except Exception:
        return None, None


# ---------------- XI ----------------
def build_xi_by_mp(df_scope: pd.DataFrame):
    result = []
    for pos in LINEUP:
        pool = df_scope[df_scope["pos"] == pos].copy()
        pool["mp"] = pd.to_numeric(pool["mp"], errors="coerce").fillna(0)
        pool["rating"] = pd.to_numeric(pool["rating"], errors="coerce")
        pool = pool.sort_values(["mp", "rating", "name"], ascending=[False, False, True])

        already = sum(1 for r in result if r["pos"] == pos)
        if already >= (2 if pos == "stp" else 1):
            continue

        pick = pool.iloc[already] if len(pool) > already else None
        if pick is None:
            result.append({"pos": pos, "name": "", "mp": None, "rating": None})
        else:
            result.append({"pos": pos, "name": pick["name"], "mp": int(pick["mp"]),
                           "rating": float(pick["rating"]) if pd.notna(pick["rating"]) else None})
    return pd.DataFrame(result)


def build_xi_by_rating(df_scope: pd.DataFrame, mp_offset: int):
    mp_series = pd.to_numeric(df_scope["mp"], errors="coerce").dropna()
    avg_mp = float(mp_series[mp_series > 0].mean()) if (mp_series > 0).any() else 0.0
    threshold = max(0.0, avg_mp + float(mp_offset))

    eligible = df_scope[pd.to_numeric(df_scope["mp"], errors="coerce").fillna(0) > threshold].copy()
    eligible["rating"] = pd.to_numeric(eligible["rating"], errors="coerce")

    result = []
    for pos in LINEUP:
        pool = eligible[eligible["pos"] == pos].sort_values(["rating", "mp", "name"], ascending=[False, False, True])
        already = sum(1 for r in result if r["pos"] == pos)
        if already >= (2 if pos == "stp" else 1):
            continue
        pick = pool.iloc[already] if len(pool) > already else None
        if pick is None:
            result.append({"pos": pos, "name": "", "rating": None, "mp": None})
        else:
            result.append({"pos": pos, "name": pick["name"],
                           "rating": float(pick["rating"]) if pd.notna(pick["rating"]) else None,
                           "mp": int(pick["mp"])})
    return pd.DataFrame(result), avg_mp, threshold


# ---------------- Pitch render ----------------
def render_pitch_xi(df_xi: pd.DataFrame, mode: str):
    if df_xi is None or df_xi.empty:
        st.info("XI yok.")
        return

    players = df_xi.to_dict("records")
    slot_map = ["GK", "CB1", "CB2", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "ST"]
    slot_data = {s: p for s, p in zip(slot_map, players)}

    def fmt2(x):
        try:
            return "—" if pd.isna(x) else f"{float(x):.2f}"
        except Exception:
            return "—"

    def card_html(slot, label_top):
        p = slot_data.get(slot, {})
        name = (p.get("Oyuncu") or "").strip()
        pos = (p.get("Poz") or "").strip()

        mp_txt = p.get("MP", None)
        mp_txt = "—" if mp_txt in [None, ""] else str(mp_txt)

        rt_txt = fmt2(p.get("Rating", None)) if p.get("Rating", None) not in [None, ""] else "—"

        if mode == "MP":
            pill1 = f"MP: {mp_txt}"
            pill2 = f"RT: {rt_txt}"
        else:
            pill1 = f"RT: {fmt2(p.get('Rating', None))}"
            pill2 = f"MP: {mp_txt}"

        return f"""
        <div class="xi-card">
          <div class="xi-top">{label_top}</div>
          <div class="xi-name">{name if name else "—"}</div>
          <div class="xi-sub">{pos.upper() if pos else ""}</div>
          <div class="xi-metrics">
            <span class="xi-pill">{pill1}</span>
            <span class="xi-pill">{pill2}</span>
          </div>
        </div>
        """

    pitch = f"""
    <style>
      .pitch-wrap {{
        position: relative;
        background: {FENER_GREEN_PITCH};
        border: 2px solid {FENER_GREEN_LINE};
        border-radius: 18px;
        padding: 16px;
        overflow: hidden;
        font-family: inherit;
      }}

      .pitch-lines {{
        position: absolute; inset: 14px;
        border: 2px solid {FENER_GREEN_LINE};
        border-radius: 14px;
        pointer-events: none;
      }}
      .half-line {{
        position: absolute;
        left: 14px; right: 14px;
        top: 50%; transform: translateY(-50%);
        border-top: 2px solid {FENER_GREEN_LINE};
        pointer-events: none;
      }}
      .center-circle {{
        position: absolute;
        left: 50%; top: 50%;
        width: 140px; height: 140px;
        transform: translate(-50%, -50%);
        border: 2px solid {FENER_GREEN_LINE};
        border-radius: 999px;
        pointer-events: none;
      }}
      .center-spot {{
        position: absolute;
        left: 50%; top: 50%;
        width: 8px; height: 8px;
        transform: translate(-50%, -50%);
        background: {FENER_GREEN_LINE};
        border-radius: 999px;
        pointer-events: none;
      }}

      .box-top {{
        position: absolute;
        left: 50%; top: 14px;
        width: 62%; height: 22%;
        transform: translateX(-50%);
        border: 2px solid {FENER_GREEN_LINE};
        pointer-events: none;
      }}
      .box-bottom {{
        position: absolute;
        left: 50%; bottom: 14px;
        width: 62%; height: 22%;
        transform: translateX(-50%);
        border: 2px solid {FENER_GREEN_LINE};
        pointer-events: none;
      }}
      .six-top {{
        position: absolute;
        left: 50%; top: 14px;
        width: 36%; height: 11%;
        transform: translateX(-50%);
        border: 2px solid {FENER_GREEN_LINE};
        pointer-events: none;
      }}
      .six-bottom {{
        position: absolute;
        left: 50%; bottom: 14px;
        width: 36%; height: 11%;
        transform: translateX(-50%);
        border: 2px solid {FENER_GREEN_LINE};
        pointer-events: none;
      }}
      .pen-top {{
        position: absolute;
        left: 50%; top: 28%;
        width: 8px; height: 8px;
        transform: translateX(-50%);
        background: {FENER_GREEN_LINE};
        border-radius: 999px;
        pointer-events: none;
      }}
      .pen-bottom {{
        position: absolute;
        left: 50%; bottom: 28%;
        width: 8px; height: 8px;
        transform: translateX(-50%);
        background: {FENER_GREEN_LINE};
        border-radius: 999px;
        pointer-events: none;
      }}

      .pitch-grid {{
        position: relative;
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        grid-template-rows: repeat(6, 92px);
        gap: 10px;
        z-index: 2;
      }}

      .xi-card {{
        background: rgba(11, 31, 58, 0.92);
        border: 1px solid {FENER_BORDER};
        border-radius: 14px;
        padding: 8px 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        font-family: inherit;
      }}
      .xi-top {{ font-size: 12px; color: {FENER_YELLOW}; font-weight: 900; }}
      .xi-name {{ font-size: 14px; font-weight: 900; color: {FENER_YELLOW}; line-height: 1.1; }}
      .xi-sub {{ font-size: 12px; color: {FENER_YELLOW}; opacity: 0.95; }}
      .xi-metrics {{ margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; }}
      .xi-pill {{
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 999px;
        border: 1px solid {FENER_BORDER};
        color: {FENER_YELLOW};
        font-family: inherit;
      }}

      .GK {{ grid-column: 3; grid-row: 6; }}
      .LB {{ grid-column: 1; grid-row: 5; }}
      .CB1 {{ grid-column: 2; grid-row: 5; }}
      .CB2 {{ grid-column: 4; grid-row: 5; }}
      .RB {{ grid-column: 5; grid-row: 5; }}

      .DM {{ grid-column: 2; grid-row: 4; }}
      .CM {{ grid-column: 4; grid-row: 4; }}
      .AM {{ grid-column: 3; grid-row: 3; }}

      .LW {{ grid-column: 2; grid-row: 2; }}
      .RW {{ grid-column: 4; grid-row: 2; }}

      /* SF 1 hücre önde */
      .ST {{ grid-column: 3; grid-row: 1; }}
    </style>

    <div class="pitch-wrap">
      <div class="pitch-lines"></div>
      <div class="half-line"></div>
      <div class="center-circle"></div>
      <div class="center-spot"></div>

      <div class="box-top"></div>
      <div class="box-bottom"></div>
      <div class="six-top"></div>
      <div class="six-bottom"></div>
      <div class="pen-top"></div>
      <div class="pen-bottom"></div>

      <div class="pitch-grid">
        <div class="ST">{card_html("ST","SF")}</div>
        <div class="LW">{card_html("LW","SLA")}</div>
        <div class="RW">{card_html("RW","SĞA")}</div>

        <div class="AM">{card_html("AM","OOS")}</div>

        <div class="DM">{card_html("DM","DOS")}</div>
        <div class="CM">{card_html("CM","GÖ")}</div>

        <div class="LB">{card_html("LB","SLB")}</div>
        <div class="CB1">{card_html("CB1","STP")}</div>
        <div class="CB2">{card_html("CB2","STP")}</div>
        <div class="RB">{card_html("RB","SĞB")}</div>

        <div class="GK">{card_html("GK","KL")}</div>
      </div>
    </div>
    """
    components.html(pitch, height=690, scrolling=False)


# ---------------- UI ----------------
st.set_page_config(page_title="Fenerbahçe PES Kariyer Paneli", layout="wide")

st.markdown(f"""
<style>
section[data-testid="stSidebar"] {{ display:none !important; }}
div[data-testid="collapsedControl"] {{ display:none !important; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {{
  background-color: {FENER_BG} !important;
}}

h1, h2, h3, h4, h5, h6, label, span, div, p {{
  color: {FENER_YELLOW} !important;
}}

input, textarea {{
  font-size: 18px !important;
  background-color: {FENER_PANEL} !important;
  color: {FENER_YELLOW} !important;
  border: 1px solid {FENER_BORDER} !important;
}}

.stButton > button {{
  background-color: {FENER_PANEL} !important;
  color: {FENER_YELLOW} !important;
  border: 2px solid {FENER_BORDER} !important;
  padding: 0.75rem 1rem !important;
  font-size: 18px !important;
  font-weight: 900 !important;
  border-radius: 14px !important;
}}
.stButton > button:hover {{
  background-color: {FENER_YELLOW} !important;
  color: {FENER_BG} !important;
}}

div[data-testid="stMainBlockContainer"] > div:first-child {{
  position: sticky;
  top: 0;
  z-index: 999;
  padding: 0.5rem 0;
  border-bottom: 2px solid {FENER_BORDER};
  background: {FENER_BG};
}}
</style>
""", unsafe_allow_html=True)

init_db()

st.write("Postgres aktif mi?", using_postgres())
if using_postgres():
    try:
        c = pg_read("SELECT COUNT(*) AS cnt FROM players")
        st.write("players count:", int(c.iloc[0]["cnt"]))
    except Exception as e:
        st.error(e)

if "page" not in st.session_state:
    st.session_state["page"] = PAGES["Giriş"]


def nav_bar(active_page: str):
    cols = st.columns([0.6, 1, 1, 1, 1, 1, 1])
    with cols[0]:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=46)

    keys = list(PAGES.keys())
    for i, k in enumerate(keys, start=1):
        target = PAGES[k]
        is_active = (active_page == target)
        with cols[i]:
            if st.button(
                k,
                use_container_width=True,
                key=f"nav_{k}",
                disabled=is_active,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["page"] = target
                st.rerun()


page = st.session_state["page"]
nav_bar(page)
st.markdown("---")

players_df = get_players_df()


def stat_form(defaults=None, key_prefix=""):
    defaults = defaults or {}
    d_mp = int(defaults.get("mp", 0) or 0)
    d_g = int(defaults.get("goals", 0) or 0)
    d_a = int(defaults.get("assists", 0) or 0)
    d_r = float(defaults.get("rating", 6.25) or 6.25)

    col1, col2, col3 = st.columns(3)
    with col1:
        season = st.selectbox("Sezon", SEASONS, index=0, key=f"{key_prefix}season")
        tournament = st.selectbox("Turnuva", TOURNAMENTS, key=f"{key_prefix}tournament")
        mp = st.number_input("Maç (MP)", min_value=0, max_value=200, value=d_mp, step=1, key=f"{key_prefix}mp")
    with col2:
        goals = st.number_input("Gol", min_value=0, max_value=300, value=d_g, step=1, key=f"{key_prefix}goals")
        assists = st.number_input("Asist", min_value=0, max_value=300, value=d_a, step=1, key=f"{key_prefix}assists")
    with col3:
        rating = st.number_input(
            "Rating",
            min_value=0.0, max_value=10.0,
            value=d_r, step=0.05, format="%.2f",
            key=f"{key_prefix}rating"
        )
    return season, tournament, mp, goals, assists, rating


# ---------------- 0) HOME ----------------
if page.startswith("0)"):
    st.title("Fenerbahçe PES Kariyer Paneli")

# ---------------- 1) NEW PLAYER ----------------
elif page.startswith("1)"):
    st.title("Yeni Oyuncu Ekle")

    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Oyuncu ismi", placeholder="Örn: Arda Güler")
    with c2:
        pos = st.selectbox("Pozisyon", POSITIONS)

    st.markdown("### İlk sezon/turnuva istatistiği")
    season, tournament, mp, goals, assists, rating = stat_form(defaults=None, key_prefix="p1_")

    if st.button("Kaydet", type="primary"):
        if not name.strip():
            st.error("İsim boş olmasın.")
        else:
            upsert_player(name.strip(), pos)
            players_now = get_players_df()
            pid = int(players_now.loc[players_now["name"] == name.strip(), "player_id"].iloc[0])

            upsert_stat(pid, season, tournament, int(mp), int(goals), int(assists), float(rating))
            recompute_general_for_player_season(pid, season)
            st.success("Kaydedildi.")

# ---------------- 2) EXISTING PLAYER ----------------
elif page.startswith("2)"):
    st.title("Mevcut Oyuncuya Giriş")

    if players_df.empty:
        st.info("Önce oyuncu ekle.")
    else:
        if "selected_player_pid" not in st.session_state:
            st.session_state["selected_player_pid"] = None
            st.session_state["selected_player_name"] = None

        if st.session_state["selected_player_pid"] is None:
            player_name, pid = pick_player_searchbox(players_df, "Oyuncu seç", key="sb_pick_player_2")
            if player_name and pid:
                st.session_state["selected_player_pid"] = pid
                st.session_state["selected_player_name"] = player_name
                st.rerun()
            st.stop()

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### Seçili Oyuncu: **{st.session_state['selected_player_name']}**")
        with c2:
            if st.button("Değiştir"):
                st.session_state["selected_player_pid"] = None
                st.session_state["selected_player_name"] = None
                clear_searchbox_state("sb_pick_player_2")
                # form state temizle
                for k in ["p2_mp", "p2_goals", "p2_assists", "p2_rating", "p2_season", "p2_tournament"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        pid = int(st.session_state["selected_player_pid"])

        # Form (sezon+turnuva seç, sonra DB kontrol)
        season, tournament, mp, goals, assists, rating = stat_form(defaults=None, key_prefix="p2_")
        existing = fetch_one_stat(pid, season, tournament)

        st.markdown("---")
        if existing:
            st.markdown("### Mevcut kayıt bulundu ✅")
            rt = existing.get("rating", None)
            rt_txt = "—" if rt is None else f"{float(rt):.2f}"
            st.markdown(
                f"**MP:** {int(existing.get('mp', 0) or 0)} | "
                f"**Gol:** {int(existing.get('goals', 0) or 0)} | "
                f"**Asist:** {int(existing.get('assists', 0) or 0)} | "
                f"**Rating:** {rt_txt}"
            )

            if st.button("Mevcut kaydı forma yükle"):
                st.session_state["p2_mp"] = int(existing.get("mp", 0) or 0)
                st.session_state["p2_goals"] = int(existing.get("goals", 0) or 0)
                st.session_state["p2_assists"] = int(existing.get("assists", 0) or 0)
                st.session_state["p2_rating"] = float(existing.get("rating", 6.25) or 6.25)
                st.rerun()

            st.info("Üstteki formu güncelleyip 'Kaydet / Güncelle' diyebilirsin.")
        else:
            st.warning("Bu sezon + turnuva için kayıt yok. Yeni giriş yapıyorsun.")

        if st.button("Kaydet / Güncelle", type="primary"):
            upsert_stat(pid, season, tournament, int(mp), int(goals), int(assists), float(rating))
            recompute_general_for_player_season(pid, season)
            st.success("Güncellendi.")

# ---------------- 3) STATS ----------------
elif page.startswith("3)"):
    st.title("İstatistikler")

    top1, top2, top3, top4 = st.columns([1, 1, 1, 1])
    with top1:
        season_filter = st.selectbox("Sezon filtresi", ["Hepsi"] + SEASONS, index=0)
    with top2:
        scope = st.selectbox("Görünüm", SCOPES, index=0)
    with top3:
        mp_offset = st.slider("Rating XI MP eşiği (Ortalama +)", min_value=-10, max_value=10, value=-2, step=1)
    with top4:
        min_mp = st.slider("Minimum MP (filtre)", min_value=0, max_value=60, value=0, step=1)

    df_main = fetch_stats_agg(scope) if season_filter == "Hepsi" else fetch_stats(season_filter, scope)

    if df_main.empty:
        st.warning("Bu filtrede data yok.")
    else:
        df_main = compute_katki(df_main)
        df_main = apply_min_mp(df_main, min_mp)

        avg_mp_preview = avg_mp_of_df(df_main) if (df_main is not None and not df_main.empty) else 0.0
        threshold_preview = max(0.0, avg_mp_preview + float(mp_offset))

        st.markdown(
            f"""
            <div style="
              background:{FENER_PANEL};
              border:1px solid {FENER_BORDER};
              border-radius:14px;
              padding:10px 12px;
              font-weight:900;
              ">
              Rating XI filtresi → Ortalama MP: {avg_mp_preview:.2f} | Eşik: {threshold_preview:.2f} | Min MP: {min_mp}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Totals (Oyuncu bazlı)")
        show = df_main[["name", "pos", "mp", "goals", "assists", "gol_asist", "rating", "katki"]].copy()
        show.rename(columns={
            "name": "Oyuncu", "pos": "Poz", "mp": "MP",
            "goals": "Gol", "assists": "Asist", "gol_asist": "Gol+Asist",
            "rating": "Rating", "katki": "Katkı Skoru"
        }, inplace=True)
        show_df(show)

        st.markdown("### XI'ler")
        tab_season, tab_career = st.tabs(["Sezon XI", "Kariyer XI"])

        def xi_block(df_xi_base: pd.DataFrame, title_left: str, title_right: str):
            l, r = st.columns(2)
            with l:
                st.markdown(f"#### {title_left}")
                xi_mp = build_xi_by_mp(df_xi_base)
                xi_mp.columns = ["Poz", "Oyuncu", "MP", "Rating"]
                show_df(xi_mp)
                render_pitch_xi(xi_mp, mode="MP")
            with r:
                st.markdown(f"#### {title_right}")
                xi_rt, _, _ = build_xi_by_rating(df_xi_base, mp_offset)
                xi_rt.columns = ["Poz", "Oyuncu", "Rating", "MP"]
                show_df(xi_rt)
                render_pitch_xi(xi_rt, mode="Rating")

        with tab_season:
            season_for_xi = season_filter
            if season_filter == "Hepsi":
                season_for_xi = st.selectbox("Sezon seç", SEASONS, index=0)

            df_season_xi = fetch_stats(season_for_xi, scope)
            df_season_xi = compute_katki(df_season_xi)
            df_season_xi = apply_min_mp(df_season_xi, min_mp)

            if df_season_xi.empty:
                st.info("Sezon XI çıkmıyor.")
            else:
                xi_block(df_season_xi, "En çok maç oynayan 11", "En yüksek ratingli 11")

        with tab_career:
            df_career_xi = fetch_stats_agg(scope)
            df_career_xi = compute_katki(df_career_xi)
            df_career_xi = apply_min_mp(df_career_xi, min_mp)

            if df_career_xi.empty:
                st.info("Kariyer XI çıkmıyor.")
            else:
                xi_block(df_career_xi, "En çok maç oynayan 11 (Kariyer)", "En yüksek ratingli 11 (Kariyer)")

        st.markdown("---")
        st.markdown("### Sezonu Komple Sil (Oyuncu + Sezon)")

        if players_df.empty:
            st.info("Oyuncu yok.")
        else:
            if "del_player_pid" not in st.session_state:
                st.session_state["del_player_pid"] = None
                st.session_state["del_player_name"] = None

            if st.session_state["del_player_pid"] is None:
                del_name, del_pid = pick_player_searchbox(players_df, "Silinecek oyuncu", key="sb_pick_player_del")
                if del_name and del_pid:
                    st.session_state["del_player_pid"] = del_pid
                    st.session_state["del_player_name"] = del_name
                    st.rerun()
                st.stop()

            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"### Seçili Oyuncu: **{st.session_state['del_player_name']}**")
            with c2:
                if st.button("Değiştir (Silme)"):
                    st.session_state["del_player_pid"] = None
                    st.session_state["del_player_name"] = None
                    clear_searchbox_state("sb_pick_player_del")
                    st.rerun()

            del_season = st.selectbox("Sezon", SEASONS, key="del_season")
            if st.button("⚠️ Bu Sezonu Komple Sil", type="secondary"):
                delete_player_season(int(st.session_state["del_player_pid"]), del_season)
                st.success("Silindi.")
                st.rerun()

# ---------------- 4) PLAYER FORM ----------------
elif page.startswith("4)"):
    st.title("Oyuncu Form")

    if players_df.empty:
        st.info("Oyuncu yok.")
    else:
        if "form_player_pid" not in st.session_state:
            st.session_state["form_player_pid"] = None
            st.session_state["form_player_name"] = None

        if st.session_state["form_player_pid"] is None:
            pn, pid = pick_player_searchbox(players_df, "Oyuncu seç", key="sb_pick_player_form")
            if pn and pid:
                st.session_state["form_player_pid"] = pid
                st.session_state["form_player_name"] = pn
                st.rerun()
            st.stop()

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"## {st.session_state['form_player_name']}")
        with c2:
            if st.button("Değiştir (Form)"):
                st.session_state["form_player_pid"] = None
                st.session_state["form_player_name"] = None
                clear_searchbox_state("sb_pick_player_form")
                st.rerun()

        pid = int(st.session_state["form_player_pid"])

        tab_genel, tab_lig, tab_kupa, tab_avrupa = st.tabs(["Genel", "Lig", "Kupa", "Avrupa"])

        def render_form_tab(scope_: str):
            df_s = fetch_player_scope_season_rows(pid, scope_)
            if df_s.empty:
                st.info("Bu scope için kayıt yok.")
                return

            df_s = add_trend_cols(df_s)
            label, diff = form_label_from_last3(df_s)

            last_season = df_s.iloc[-1]["season"]
            last_rating = df_s.iloc[-1]["rating"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Form", label, f"{diff:+.2f}" if diff is not None else None)
            m2.metric("Son Sezon", str(last_season))
            m3.metric("Son Rating", f"{float(last_rating):.2f}" if pd.notna(last_rating) else "—")

            out = df_s[["season", "mp", "goals", "assists", "gol_asist", "rating",
                        "ΔMP", "ΔGol", "ΔAsist", "ΔGol+Asist", "ΔRating"]].copy()
            out.rename(columns={
                "season": "Sezon", "mp": "MP", "goals": "Gol", "assists": "Asist",
                "gol_asist": "Gol+Asist", "rating": "Rating",
            }, inplace=True)
            show_df(out)

        with tab_genel:
            render_form_tab("Genel")
        with tab_lig:
            render_form_tab("Lig")
        with tab_kupa:
            render_form_tab("Kupa")
        with tab_avrupa:
            render_form_tab("Avrupa")

# ---------------- 5) RECORDS ----------------
else:
    st.title("Rekorlar")

    c1, c2, c3 = st.columns(3)
    with c1:
        scope = st.selectbox("Scope", SCOPES, index=0)
    with c2:
        min_mp = st.slider("Minimum MP (rating listeleri)", min_value=0, max_value=60, value=0, step=1)
    with c3:
        rating_mp_offset = st.slider("Rating MP eşiği (Ortalama +)", min_value=-10, max_value=10, value=0, step=1)

    df_career = fetch_stats_agg(scope)
    df_career = compute_katki(df_career)

    if df_career.empty:
        st.info("Bu scope için data yok.")
    else:
        st.markdown("### Top 10 (Kariyer)")
        metric_pick_c = st.selectbox("Metriği seç", ["Gol", "Asist", "Gol+Asist", "Katkı Skoru", "Rating"], index=3)

        df_top = df_career.copy()
        df_top["rating"] = pd.to_numeric(df_top["rating"], errors="coerce").round(2)

        if metric_pick_c == "Rating":
            avg_mp = avg_mp_of_df(df_top)
            threshold = max(0.0, avg_mp + float(rating_mp_offset))
            df_top = apply_min_mp(df_top, min_mp)
            df_top = df_top[pd.to_numeric(df_top["mp"], errors="coerce").fillna(0) > threshold]
            df_top = df_top.sort_values(["rating", "mp"], ascending=[False, False])
            cols = ["name", "pos", "mp", "rating"]
        elif metric_pick_c == "Gol":
            df_top = df_top.sort_values(["goals", "mp"], ascending=[False, False])
            cols = ["name", "pos", "mp", "goals", "assists", "gol_asist", "rating"]
        elif metric_pick_c == "Asist":
            df_top = df_top.sort_values(["assists", "mp"], ascending=[False, False])
            cols = ["name", "pos", "mp", "assists", "goals", "gol_asist", "rating"]
        elif metric_pick_c == "Gol+Asist":
            df_top = df_top.sort_values(["gol_asist", "mp"], ascending=[False, False])
            cols = ["name", "pos", "mp", "gol_asist", "goals", "assists", "rating"]
        else:
            df_top = df_top.sort_values(["katki", "mp"], ascending=[False, False])
            cols = ["name", "pos", "mp", "katki", "rating", "gol_asist"]

        df_top10 = df_top[cols].head(10).copy()
        df_top10.rename(columns={
            "name": "Oyuncu", "pos": "Poz", "mp": "MP",
            "goals": "Gol", "assists": "Asist", "gol_asist": "Gol+Asist",
            "rating": "Rating", "katki": "Katkı Skoru"
        }, inplace=True)
        show_df(df_top10)

        st.markdown("---")
        st.markdown("### Top 10 (Tek Sezon)")
        df_season = fetch_season_records(scope)
        df_season = compute_katki(df_season)

        metric_pick_s = st.selectbox("Metriği seç (Tek Sezon)", ["Gol", "Asist", "Gol+Asist", "Katkı Skoru", "Rating"], index=3)

        df_ts = df_season.copy()
        df_ts["rating"] = pd.to_numeric(df_ts["rating"], errors="coerce").round(2)

        if metric_pick_s == "Rating":
            avg_mp = avg_mp_of_df(df_ts)
            threshold = max(0.0, avg_mp + float(rating_mp_offset))
            df_ts = apply_min_mp(df_ts, min_mp)
            df_ts = df_ts[pd.to_numeric(df_ts["mp"], errors="coerce").fillna(0) > threshold]
            df_ts = df_ts.sort_values(["rating", "mp"], ascending=[False, False])
            cols = ["season", "name", "pos", "mp", "rating"]
        elif metric_pick_s == "Gol":
            df_ts = df_ts.sort_values(["goals", "mp"], ascending=[False, False])
            cols = ["season", "name", "pos", "mp", "goals", "assists", "gol_asist", "rating"]
        elif metric_pick_s == "Asist":
            df_ts = df_ts.sort_values(["assists", "mp"], ascending=[False, False])
            cols = ["season", "name", "pos", "mp", "assists", "goals", "gol_asist", "rating"]
        elif metric_pick_s == "Gol+Asist":
            df_ts = df_ts.sort_values(["gol_asist", "mp"], ascending=[False, False])
            cols = ["season", "name", "pos", "mp", "gol_asist", "goals", "assists", "rating"]
        else:
            df_ts = df_ts.sort_values(["katki", "mp"], ascending=[False, False])
            cols = ["season", "name", "pos", "mp", "katki", "rating", "gol_asist"]

        df_ts10 = df_ts[cols].head(10).copy()
        df_ts10.rename(columns={
            "season": "Sezon",
            "name": "Oyuncu", "pos": "Poz", "mp": "MP",
            "goals": "Gol", "assists": "Asist", "gol_asist": "Gol+Asist",
            "rating": "Rating", "katki": "Katkı Skoru"
        }, inplace=True)
        show_df(df_ts10)

