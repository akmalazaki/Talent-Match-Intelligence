# Talent Match Intelligence — Succession Dashboard (Step 3)
# Run with: streamlit run "Talent Match Intelligent Dashboard.py"

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv, find_dotenv
import plotly.express as px
import plotly.graph_objects as go
import requests

# =========================
# Setup & Connection
# =========================
load_dotenv(find_dotenv())

DB_CONFIG = {
    "USER": os.getenv("user"),
    "PASSWORD": os.getenv("password"),
    "HOST": os.getenv("host"),
    "PORT": os.getenv("port", "5432"),
    "DBNAME": os.getenv("dbname")
}

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "HTTP-Referer": "http://localhost",
}

@st.cache_resource(show_spinner=False)
def get_engine():
    try:
        return create_engine(
            f"postgresql://{DB_CONFIG['USER']}:{DB_CONFIG['PASSWORD']}@"
            f"{DB_CONFIG['HOST']}:{DB_CONFIG['PORT']}/{DB_CONFIG['DBNAME']}"
        )
    except Exception as e:
        st.error(f"Gagal koneksi DB: {e}")
        return None

engine = get_engine()
if engine is None:
    st.stop()

# =========================
# SQL Loader
# =========================
def sql_df(query: str, params=None):
    try:
        return pd.read_sql(text(query), engine, params=params)
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return pd.DataFrame()

# ===== Theme helpers (Corporate Blue) =====
BLUE_1 = "#0ea5e9"  # light blue
BLUE_2 = "#2563eb"  # primary blue
BLUE_3 = "#1e40af"  # deep blue
BLUE_4 = "#0b1b3b"  # navy

def readiness_label(score: float) -> str:
    if pd.isna(score):
        return "Development Priority"
    if score >= 105:
        return "Strong Successor"
    elif score >= 95:
        return "Ready Soon"
    else:
        return "Development Priority"

def readiness_color(label: str) -> str:
    # Corporate blue palette instead of traffic light
    return {
        "Strong Successor": BLUE_2,
        "Ready Soon": BLUE_1,
        "Development Priority": BLUE_3,  # darker blue
    }.get(label, BLUE_1)

# =========================
# Core SQL (Templated)
# =========================
SQL_BASE = """
with benchmark as (
    select 
        c.pillar_code,
        avg(c.score::numeric) FILTER (WHERE c.score IS NOT NULL AND c.score <> 'NaN') as baseline_score
    from competencies_yearly c
    where c.year = :yr
      and c.employee_id = any(:hp_ids)
      and c.score <= 5
    group by c.pillar_code
),
emp_comp as (
    select 
        c.employee_id,
        c.pillar_code,
        avg(c.score::numeric) as user_score
    from competencies_yearly c
    join performance_yearly pr 
      on pr.employee_id = c.employee_id 
     and pr.year = c.year
    where c.year = :yr
      and pr.rating between 1 and 5
      and c.score <= 5
    group by c.employee_id, c.pillar_code
),
tv_match as (
    select 
        e.employee_id,
        e.pillar_code as tv_name,
        (e.user_score / b.baseline_score) * 100 as tv_match_rate,
        b.baseline_score,
        e.user_score
    from emp_comp e
    join benchmark b on b.pillar_code = e.pillar_code
),
tgv as (
    select 
        employee_id,
        avg(case when tv_name in ('QDD','FTC','SEA') then tv_match_rate end) as exec_match_rate,
        avg(case when tv_name in ('STO','VCU','IDS','CEX') then tv_match_rate end) as strat_match_rate,
        avg(case when tv_name in ('CSI','GDR','LIE') then tv_match_rate end) as lead_match_rate
    from tv_match
    group by employee_id
),
final_match as (
    select
        employee_id,
        exec_match_rate,
        strat_match_rate,
        lead_match_rate,
        (
            coalesce(exec_match_rate, 0) * 0.40 +
            coalesce(strat_match_rate, 0) * 0.35 +
            coalesce(lead_match_rate, 0) * 0.25
        ) as final_match_rate
    from tgv
)
"""

SQL_EMP_SUMMARY = SQL_BASE + """
select 
    f.employee_id,
    e.fullname,
    e.grade_id,
    e.directorate_id,
    e.department_id,
    e.position_id,
    e.years_of_service_months,
    f.exec_match_rate,
    f.strat_match_rate,
    f.lead_match_rate,
    f.final_match_rate
from final_match f
join employees e using (employee_id)
where f.final_match_rate is not null
order by f.final_match_rate desc, f.employee_id;
"""

SQL_TV_DETAIL = SQL_BASE + """
select 
    f.employee_id,
    tm.tv_name,
    case 
        when tm.tv_name in ('QDD','FTC','SEA') then 'Execution Excellence'
        when tm.tv_name in ('STO','VCU','IDS','CEX') then 'Strategic Value'
        when tm.tv_name in ('CSI','GDR','LIE') then 'Leadership & Influence'
    end as tgv_name,
    tm.baseline_score,
    tm.user_score,
    tm.tv_match_rate,
    case 
        when tm.tv_name in ('QDD','FTC','SEA') then f.exec_match_rate
        when tm.tv_name in ('STO','VCU','IDS','CEX') then f.strat_match_rate
        when tm.tv_name in ('CSI','GDR','LIE') then f.lead_match_rate
    end as tgv_match_rate,
    f.final_match_rate
from final_match f
left join tv_match tm using (employee_id)
where f.employee_id = :emp
order by tm.tv_name;
"""

# =========================
# High Performer Selector
# =========================
st.title("Talent Match Intelligence — Succession Dashboard")

st.sidebar.header("Define Vacancy Benchmark")
role_name = st.sidebar.text_input("Role Name", placeholder="e.g., Data Analyst")
job_level = st.sidebar.selectbox("Job Level (grade)", list(range(1, 6)), index=3)
role_purpose = st.sidebar.text_area("Role Purpose (short)")
YEAR = st.sidebar.number_input("Year", 2000, 2100, 2024, step=1)

hp_df = sql_df("""
select employee_id, fullname from employees e
join performance_yearly p using(employee_id)
where p.year = :yr and p.rating = 5
order by fullname
""", {"yr": YEAR})

if hp_df.empty:
    st.warning("Tidak ada High Performer pada tahun ini.")
    st.stop()

hp_df["label"] = hp_df["fullname"] + " — " + hp_df["employee_id"]
selected_hp = st.sidebar.multiselect(
    "Select Benchmark Employees", hp_df["label"].tolist()
)
label_to_id = dict(zip(hp_df["label"], hp_df["employee_id"]))
hp_ids = [label_to_id.get(x) for x in selected_hp] or hp_df["employee_id"].tolist()

submitted = st.sidebar.button("Generate Matching Results")

if not submitted:
    st.info("Isi benchmark dan klik Generate Matching Results.")
    st.stop()

# =========================
# Compute Matching
# =========================
df_rank = sql_df(SQL_EMP_SUMMARY, {"yr": YEAR, "hp_ids": hp_ids})
if df_rank.empty:
    st.warning("Tidak ada hasil. Sesuaikan kriteria.")
    st.stop()

# =========================
# AI Job Profile
# =========================
st.subheader("💡 AI-Generated Job Profile")
prompt = f"""
Buat job profile untuk:
Role: {role_name}
Level: {job_level}
Purpose: {role_purpose}

Format:
1. Key Responsibilities
2. Job Requirements
3. Key Competencies
"""

with st.spinner("AI generating job profile..."):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {"role": "system", "content": "You are an HR expert."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=20
        )
        st.write(response.json()["choices"][0]["message"]["content"])
    except Exception:
        st.error("Gagal memanggil OpenRouter")

# =========================
# Ranking Table + Readiness
# =========================
st.subheader("🏆 Ranked Candidates (Final Match Rate)")

# Tambah readiness lebih dulu agar tersedia untuk df_show
df_rank["readiness"] = df_rank["final_match_rate"].apply(readiness_label)

df_show = df_rank.copy()
for c in ["final_match_rate","exec_match_rate","strat_match_rate","lead_match_rate"]:
    df_show[c] = df_show[c].round(2)

# urutkan dan tampilkan kolom penting (termasuk readiness)
df_show = df_show[[
    "employee_id","fullname","grade_id","directorate_id","department_id",
    "position_id","years_of_service_months",
    "exec_match_rate","strat_match_rate","lead_match_rate","final_match_rate","readiness"
]]

st.dataframe(df_show, use_container_width=True, hide_index=True)

# Export shortlist
csv = df_show.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Shortlist (CSV)",
    data=csv,
    file_name="succession_shortlist.csv",
    mime="text/csv"
)

st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)

# ===== Readiness label + summary =====
summary = (
    df_rank["readiness"]
    .value_counts()
    .reindex(["Strong Successor", "Ready Soon", "Development Priority"])
    .fillna(0)
    .astype(int)
)
total = int(summary.sum()) if int(summary.sum()) > 0 else 1
pct = (summary / total * 100).round(1)

st.markdown("### 📌 Pipeline Readiness Summary")
c1, c2, c3 = st.columns(3)
for col, title in zip(
    [c1, c2, c3],
    ["Strong Successor", "Ready Soon", "Development Priority"]
):
    with col:
        st.markdown(
            f"""
            <div style="border:1px solid {readiness_color(title)}; border-radius:12px; padding:12px;">
              <div style="font-size:14px; color:{readiness_color(title)};">{title}</div>
              <div style="font-size:28px; font-weight:700; color:white;">{summary[title]} <span style="font-size:14px;">cand.</span></div>
              <div style="font-size:13px; color:#9aa4b2;">{pct[title]}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)

# =========================
# Histogram
# =========================
st.subheader("📊 Final Match Distribution")

# Buckets by readiness
bins = [0, 95, 105, df_rank["final_match_rate"].max() + 1]
labels = ["Development Priority","Ready Soon","Strong Successor"]
df_rank["readiness_bucket"] = pd.cut(
    df_rank["final_match_rate"], bins=bins, labels=labels, right=False
)

fig_hist = px.histogram(
    df_rank, x="final_match_rate", color="readiness_bucket",
    nbins=30, barmode="overlay",
    color_discrete_map={
        "Strong Successor": BLUE_2,
        "Ready Soon": BLUE_1,
        "Development Priority": BLUE_3
    },
    labels={"final_match_rate":"Final Match Rate","readiness_bucket":"Readiness"}
)
fig_hist.update_traces(opacity=0.85)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)

# =========================
# Candidate Detail
# =========================
st.subheader("🔎 Candidate TV / TGV Detail")
emp_choice = st.selectbox(
    "Pilih kandidat:",
    df_rank["employee_id"].tolist(),
    format_func=lambda x: df_rank.loc[df_rank.employee_id == x, "fullname"].iloc[0],
)

if emp_choice:
    df_tv = sql_df(SQL_TV_DETAIL, {"yr": YEAR, "hp_ids": hp_ids, "emp": emp_choice})
    if not df_tv.empty:
        # Tabel detail
        st.dataframe(df_tv.round(2), use_container_width=True)

        # Radar TV
        tv_order = ["CEX","CSI","FTC","GDR","IDS","LIE","QDD","SEA","STO","VCU"]
        df_plot = df_tv.set_index("tv_name").reindex(tv_order).dropna().reset_index()

        categories = df_plot["tv_name"].tolist()
        if categories:
            fig = go.Figure()
            for col, label in [("user_score", "User"), ("baseline_score", "Benchmark")]:
                vals_base = df_plot[col].tolist()
                vals = vals_base + [vals_base[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=label
                ))
            fig.update_layout(
                showlegend=True,
                title="Radar TV",
                polar=dict(radialaxis=dict(visible=True, range=[0, 5]))
            )
            st.plotly_chart(fig, use_container_width=True)

        # TGV Bars
        tgv_bar = df_tv.groupby("tgv_name")["tgv_match_rate"].mean().round(2).reset_index()
        st.plotly_chart(
            px.bar(tgv_bar, x="tgv_name", y="tgv_match_rate", title="TGV Match"),
            use_container_width=True
        )

        # ====== Strength Gap (TGV) Bar: candidate vs benchmark (100) ======
        st.markdown("### 📈 TGV Strength Gap (vs Benchmark = 100)")
        tgv_gap = (
            df_tv.groupby("tgv_name", as_index=False)["tgv_match_rate"]
            .mean()
            .rename(columns={"tgv_match_rate":"candidate_tgv"})
        )
        tgv_gap["benchmark"] = 100.0
        tgv_gap["gap_vs_100"] = (tgv_gap["candidate_tgv"] - 100).round(2)

        fig_gap = px.bar(
            tgv_gap, x="tgv_name", y="gap_vs_100",
            color="gap_vs_100",
            color_continuous_scale=[BLUE_3, BLUE_1, BLUE_2],
            labels={"gap_vs_100":"Gap vs 100 (%)","tgv_name":"TGV"},
            title=None
        )
        fig_gap.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_gap, use_container_width=True)

        # ====== HR Advisory Narrative (OpenRouter) ======
        st.markdown("### 🧭 HR Advisory — Narrative & Recommendation")

        top_name = df_rank.loc[df_rank["employee_id"] == emp_choice, "fullname"].iloc[0]
        fm = float(df_rank.loc[df_rank["employee_id"] == emp_choice, "final_match_rate"].iloc[0])
        red_label = readiness_label(fm)

        # Ringkas angka TGV dalam text
        tgv_pairs = [f"{r.tgv_name}: {r.candidate_tgv:.1f}%" for r in tgv_gap.itertuples(index=False)]
        tgv_str = "; ".join(tgv_pairs)

        advisory_prompt = f"""
Anda adalah HR advisor yang menulis analisis suksesi untuk atasan bisnis.

Kandidat: {top_name} ({emp_choice})
Final Match: {fm:.1f}% → {red_label}
Rerata TGV kandidat (vs benchmark 100): {tgv_str}.

Tuliskan ringkas (3 bagian) dalam Bahasa Indonesia:
1) Readiness ringkas (1–2 kalimat), jelaskan kenapa.
2) Strengths (bullet points; berdasarkan TGV yang tertinggi vs 100).
3) Development plan 3–4 butir (praktis, 6–12 bulan) sesuai gap (TGV terendah).

Nada: profesional, analitis, to-the-point, empowering.
Jangan menulis ulang angka panjang; cukup sebut domain/arah penguatan.
"""

        try:
            payload = {
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a concise HR advisor for succession planning."},
                    {"role": "user", "content": advisory_prompt}
                ],
                "temperature": 0.4,
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=30
            )
            narrative = resp.json()["choices"][0]["message"]["content"]
            st.write(narrative)
        except Exception:
            st.info("AI narrative tidak tersedia saat ini. Cek OPENROUTER_API_KEY di .env.")