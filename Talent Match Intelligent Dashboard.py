# Talent Match Intelligence — Succession Dashboard (Step 3)
# python -m streamlit run "Talent Match Intelligent Dashboard.py"

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv, find_dotenv
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit.components.v1 as components
from datetime import datetime
from fpdf import FPDF
import tempfile

# =========================
# Setup & Connection
# =========================
if os.path.exists(".env"):
    load_dotenv(find_dotenv())

# Helper untuk ambil secret dengan aman
def safe_get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

DB_CONFIG = {
    "USER": os.getenv("user") or safe_get_secret("user"),
    "PASSWORD": os.getenv("password") or safe_get_secret("password"),
    "HOST": os.getenv("host") or safe_get_secret("host"),
    "PORT": os.getenv("port") or safe_get_secret("port", "5432"),
    "DBNAME": os.getenv("dbname") or safe_get_secret("dbname"),
}

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY") or safe_get_secret("OPENROUTER_API_KEY")
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
# Helpers
# =========================
def sql_df(query: str, params=None):
    try:
        return pd.read_sql(text(query), engine, params=params)
    except Exception as e:
        st.warning(f"⚠️ SQL error terjadi: {e}")
        return pd.DataFrame()

# Convert Python list -> PostgreSQL array literal
def to_pg_array(v: list[str]) -> str:
    if not v:
        return "{}"
    return "{" + ",".join(f'"{x}"' for x in v) + "}"

def generate_pdf_report(candidate_name, emp_id, summary, radar_fig, gap_fig, narrative_text):
    """Generate temporary PDF report file and return its path."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Talent Match Intelligence - Succession Report", ln=True, align="C")
    pdf.ln(8)
    
    # Candidate info
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Candidate: {candidate_name} ({emp_id})", ln=True)
    for k, v in summary.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.ln(4)

    # --- Tambahkan konfigurasi Kaleido (tanpa Chrome dependency) ---
    pio.kaleido.scope.default_format = "png"
    pio.kaleido.scope.default_width = 800
    pio.kaleido.scope.default_height = 600
    pio.kaleido.scope.default_scale = 1.5
    
    # --- Radar chart image ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_radar:
        radar_fig.write_image(tmp_radar.name)
        radar_img_path = tmp_radar.name
    
    # --- Gap chart image ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_gap:
        gap_fig.write_image(tmp_gap.name)
        gap_img_path = tmp_gap.name

    # Insert charts
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Competency Match Overview", ln=True)
    pdf.image(radar_img_path, w=150)
    pdf.ln(5)
    pdf.image(gap_img_path, w=150)
    pdf.ln(8)

    # Narrative
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "AI Advisory Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, narrative_text)

    # Save to temporary PDF file
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name


BLUE_1 = "#0ea5e9"
BLUE_2 = "#2563eb"
BLUE_3 = "#1e40af"

def readiness_label(score: float) -> str:
    if pd.isna(score): return "Development Priority"
    if score >= 105: return "Strong Successor"
    if score >= 95: return "Ready Soon"
    return "Development Priority"

def readiness_color(label: str) -> str:
    return {
        "Strong Successor": BLUE_2,
        "Ready Soon": BLUE_1,
        "Development Priority": BLUE_3,
    }.get(label, BLUE_1)

def scroll_to(anchor):
    components.html(
        f"""
        <script>
        const go = () => {{
            window.parent.location.hash = "#{anchor}";
        }};
        setTimeout(go, 60);
        </script>
        """,
        height=0,
    )

# =========================
# SQL QUERIES (NO CAST IN SQL)
# =========================
SQL_BASE = """
with benchmark as (
    select 
        c.pillar_code,
        avg(c.score::numeric)
        FILTER (WHERE c.score IS NOT NULL AND c.score <> 'NaN')
        as baseline_score
    from competencies_yearly c
    where c.year = :yr
      and c.employee_id = ANY(:hp_ids)
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
      on pr.employee_id = c.employee_id AND pr.year = c.year
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
  and e.position_id = coalesce(:role_id, e.position_id)
  and e.grade_id = coalesce(:grade_id, e.grade_id)
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
order by tm.tv_name
"""

# Ambil daftar role (positions) dan grade (grades)
positions_df = sql_df("SELECT DISTINCT position_id, name FROM dim_positions ORDER BY name;")
grades_df = sql_df("SELECT DISTINCT grade_id, name FROM dim_grades ORDER BY name;")

st.sidebar.header("Define Vacancy Benchmark")

# Dropdown dinamis
selected_role = st.sidebar.selectbox(
    "Role (Position)",
    options=positions_df["name"].tolist(),
    index=None,
)

selected_grade = st.sidebar.selectbox(
    "Job Level (Grade)",
    options=grades_df["name"].tolist(),
    index=None
)

st.session_state.role_name = selected_role
st.session_state.job_level = selected_grade

# Ambil ID-nya untuk query SQL
# Ambil ID-nya untuk query SQL
role_id = None
grade_id = None
if selected_role:
    role_id = int(positions_df.loc[positions_df["name"] == selected_role, "position_id"].iloc[0])
if selected_grade:
    grade_id = int(grades_df.loc[grades_df["name"] == selected_grade, "grade_id"].iloc[0])

# =========================
# SESSION STATE DEFAULTS
# =========================

defaults = {
    "submitted": False,
    "role_name": selected_role,
    "job_level": selected_grade,
    "role_purpose": "",
    "year": 2024,
    "hp_ids": [],
    "selected_emp": None,
    "scroll_to_detail": False,
    "shortlists": []
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# =========================
# Sidebar Inputs
# =========================
st.markdown(
    """
    <style>
    /* Atur warna teks sesuai tema Streamlit */
    [data-theme="light"] h3.app-title { color: #111111; }  /* hitam untuk background putih */
    [data-theme="dark"] h3.app-title { color: #FFFFFF; }   /* putih untuk background gelap */

    h3.app-title {
        text-align: left;
        font-size: 1.6rem;
        margin-top: -1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    hr.app-line {
        border: 0.5px solid rgba(255,255,255,0.1);
        margin-top: -0.2rem;
    }

    [data-theme="light"] hr.app-line {
        border-color: rgba(0,0,0,0.1);
    }
    </style>

    <h3 class='app-title'>
        Talent Match Intelligence — Succession Dashboard
    </h3>
    <hr class='app-line'>
    """,
    unsafe_allow_html=True
)

def set_state(k):
    st.session_state[k] = st.session_state[f"_{k}"]

st.sidebar.text_area(
    "Role Purpose (short)",
    key="_role_purpose",
    value=st.session_state.role_purpose,
    on_change=lambda: set_state("role_purpose"),
    placeholder="e.g., Oversee data-driven decision making..."
)

st.sidebar.number_input(
    "Year", 2000, 2100,
    st.session_state.year, step=1,
    key="_year",
    on_change=lambda: set_state("year")
)

# Load HP list
hp_df = sql_df("""
select employee_id, fullname from employees e
join performance_yearly p using(employee_id)
where p.year = :yr and p.rating = 5
order by fullname
""", {"yr": st.session_state.year})

hp_df["label"] = hp_df["fullname"] + " — " + hp_df["employee_id"]
label_to_id = dict(zip(hp_df["label"], hp_df["employee_id"]))

# Benchmark selector (Kosong = fallback all HP)
selected_hp_labels = st.sidebar.multiselect(
    "Select Benchmark Employees (optional)",
    hp_df["label"].tolist()
)
if selected_hp_labels:
    st.session_state.hp_ids = [label_to_id[x] for x in selected_hp_labels]
else:
    st.session_state.hp_ids = hp_df["employee_id"].tolist()

# Generate
if st.sidebar.button("Generate Matching Results"):
    st.session_state.submitted = True

# ✅ RESET STATE BUTTON (final merged)
if st.sidebar.button("🔄 Reset State"):
    for k in list(st.session_state.keys()):
        if k not in defaults:
            del st.session_state[k]
    st.session_state.update(defaults)
    st.rerun()

if not st.session_state.submitted:
    st.info("Isi dan klik Generate Matching Results.")
    st.stop()

# =========================
# Compute & Inject Array Literal into SQL
# =========================
hp_array_literal = f"'{to_pg_array(st.session_state.hp_ids)}'"

@st.cache_data(ttl=600, show_spinner=False)
def compute_ranking(year: int, hp_array_literal: str, role_id: str, grade_id: str):
    sql_rank = SQL_EMP_SUMMARY.replace(":hp_ids", hp_array_literal)
    return sql_df(sql_rank, {"yr": year, "role_id": role_id, "grade_id": grade_id})

try:
    df_rank = compute_ranking(st.session_state.year, hp_array_literal, role_id, grade_id)
except Exception as e:
    st.error(f"Gagal menghitung ranking: {e}")
    st.stop()

if df_rank.empty:
    st.warning("Tidak ada hasil matching.")
    st.stop()

# =========================
# TABS 🎯
# =========================
tab_overview, tab_candidates, tab_detail, tab_shortlists = st.tabs(
    ["Overview", "Candidates", "Detail", "Shortlists"]
)

# =========================================
# Overview Tab
# =========================================
with tab_overview:

    # 👇 AI Job Profile (output tetap sama)
    st.subheader("💡 AI-Generated Job Profile")
    prompt = f"""
Buat job profile untuk:
Role: {selected_role}
Level: {selected_grade}
Purpose: {st.session_state.role_purpose}

Format:
1. Key Responsibilities
2. Job Requirements
3. Key Competencies
"""
    with st.spinner("Generating job profile..."):
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
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty AI response.")
            st.write(content)
        except Exception as e:
            st.error(f"Gagal memanggil AI Job Profile: {e}")

    # 📌 Pipeline Readiness Summary (logika sama)
    df_rank["readiness"] = df_rank["final_match_rate"].apply(readiness_label)
    # ===== Summary cards =====
    st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)
    st.markdown("### 📌 Pipeline Readiness Summary")

    summary = (
        df_rank["readiness"]
        .value_counts()
        .reindex(["Strong Successor", "Ready Soon", "Development Priority"])
        .fillna(0)
        .astype(int)
    )
    total = int(summary.sum()) if int(summary.sum()) > 0 else 1
    pct = (summary / total * 100).round(1)

    c1, c2, c3 = st.columns(3)
    for col, title in zip([c1, c2, c3],
                          ["Strong Successor", "Ready Soon", "Development Priority"]):
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

    # ===== Histogram =====
    st.subheader("📊 Final Match Distribution")
    bins = [0, 95, 105, df_rank["final_match_rate"].max() + 1]
    labels = ["Development Priority", "Ready Soon", "Strong Successor"]
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
        labels={"final_match_rate": "Final Match Rate", "readiness_bucket": "Readiness"}
    )
    fig_hist.update_traces(opacity=0.85)
    st.plotly_chart(fig_hist, use_container_width=True)

# =========================================
# Candidates Tab (Ranking + Export + Save shortlist)
# =========================================
with tab_candidates:
    st.subheader("🏆 Ranked Candidates (Final Match Rate)")

    df_show = df_rank.copy()
    for c in ["final_match_rate", "exec_match_rate", "strat_match_rate", "lead_match_rate"]:
        df_show[c] = df_show[c].round(2)
    df_show["readiness"] = df_show["final_match_rate"].apply(readiness_label)

    df_show = df_show[[
        "employee_id", "fullname", "grade_id", "directorate_id", "department_id",
        "position_id", "years_of_service_months",
        "exec_match_rate", "strat_match_rate", "lead_match_rate", "final_match_rate", "readiness"
    ]]
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Export CSV
    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Shortlist (CSV)",
        data=csv,
        file_name="succession_shortlist.csv",
        mime="text/csv"
    )

    # Save shortlist to session for later comparison
    from datetime import datetime
    if st.button("💾 Save this shortlist for comparison"):
        st.session_state.shortlists.append({
            "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": st.session_state.role_name or "(no-role-name)",
            "year": st.session_state.year,
            "df": df_show.copy()
        })
        st.success("Shortlist saved to session. Lihat di tab 'Shortlists'.")

# =========================================
# Detail Tab (Candidate TV/TGV + Auto-scroll)
# =========================================
with tab_detail:
    st.markdown('<a name="candidate-detail"></a>', unsafe_allow_html=True)
    st.subheader("🔎 Candidate TV / TGV Detail")

    # Select candidate (stateful) + auto-scroll trigger
    def on_change_candidate():
        st.session_state.selected_emp = st.session_state._emp_choice
        st.session_state.scroll_to_detail = True

    emp_options = df_rank["employee_id"].tolist()
    default_index = 0
    if st.session_state.selected_emp in emp_options:
        default_index = emp_options.index(st.session_state.selected_emp)

    emp_choice = st.selectbox(
        "Pilih kandidat:",
        options=emp_options,
        index=default_index,
        key="_emp_choice",
        format_func=lambda x: df_rank.loc[df_rank.employee_id == x, "fullname"].iloc[0],
        on_change=on_change_candidate
    )

    # Auto scroll after selection
    if st.session_state.scroll_to_detail:
        scroll_to("candidate-detail")
        st.session_state.scroll_to_detail = False

    if emp_choice:
        @st.cache_data(ttl=300, show_spinner=False)
        def fetch_tv_detail(year: int, hp_array_literal: str, emp: str):
            sql_tv = SQL_TV_DETAIL.replace(":hp_ids", hp_array_literal)
            return sql_df(sql_tv, {"yr": year, "emp": emp, "role_id":role_id, "grade_id":grade_id})

        with st.spinner("Fetching candidate detail..."):
            df_tv = fetch_tv_detail(
                st.session_state.year,
                hp_array_literal,
                emp_choice
            )

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

            # Strength Gap vs 100
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

            # HR Advisory Narrative (AI)
            st.markdown("### 🧭 HR Advisory — Narrative & Recommendation")
            top_name = df_rank.loc[df_rank["employee_id"] == emp_choice, "fullname"].iloc[0]
            fm = float(df_rank.loc[df_rank["employee_id"] == emp_choice, "final_match_rate"].iloc[0])
            red_label = readiness_label(fm)
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
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=HEADERS,
                    json={
                        "model": "mistralai/mistral-7b-instruct",
                        "messages": [
                            {"role": "system", "content": "You are a concise HR advisor for succession planning."},
                            {"role": "user", "content": advisory_prompt}
                        ],
                        "temperature": 0.4,
                    },
                    timeout=30
                )
                narrative = resp.json()["choices"][0]["message"]["content"]
                st.write(narrative)
            except Exception:
                st.info("AI narrative tidak tersedia saat ini. Cek OPENROUTER_API_KEY di .env.")

    # === Export PDF Report ===
    st.markdown("### 📄 Export Succession Report")

    summary_info = {
        "Exec Match": f"{df_tv['tgv_match_rate'][df_tv['tgv_name']=='Execution Excellence'].mean():.1f}%",
        "Strat Match": f"{df_tv['tgv_match_rate'][df_tv['tgv_name']=='Strategic Value'].mean():.1f}%",
        "Lead Match": f"{df_tv['tgv_match_rate'][df_tv['tgv_name']=='Leadership & Influence'].mean():.1f}%",
        "Final Match": f"{fm:.1f}%",
        "Readiness": red_label,
    }

    if st.button("📑 Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            pdf_path = generate_pdf_report(
                candidate_name=top_name,
                emp_id=emp_choice,
                summary=summary_info,
                radar_fig=fig,
                gap_fig=fig_gap,
                narrative_text=narrative if 'narrative' in locals() else "(AI narrative unavailable)"
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Succession Report (PDF)",
                    data=f,
                    file_name=f"{top_name.replace(' ', '_')}_Succession_Report.pdf",
                    mime="application/pdf"
                )


# =========================================
# Shortlists Tab (compare saved vs current)
# =========================================
with tab_shortlists:
    st.subheader("🗂 Saved Shortlists (Session)")

    if not st.session_state.shortlists:
        st.info("Belum ada shortlist yang disimpan. Simpan dari tab **Candidates**.")
    else:
        meta = [
            f"{i+1}. {s['when']} — {s['role']} (Year {s['year']})"
            for i, s in enumerate(st.session_state.shortlists)
        ]
        idx_left = st.selectbox(
            "Left shortlist", options=list(range(len(meta))),
            format_func=lambda i: meta[i]
        )
        idx_right = st.selectbox(
            "Right shortlist (optional)", options=[None] + list(range(len(meta))),
            format_func=lambda i: ("(None)" if i is None else meta[i])
        )

        colL, colR = st.columns(2)
        with colL:
            st.markdown("**Left**")
            st.dataframe(
                st.session_state.shortlists[idx_left]["df"],
                use_container_width=True, hide_index=True
            )
        with colR:
            st.markdown("**Right**")
            if idx_right is not None:
                st.dataframe(
                    st.session_state.shortlists[idx_right]["df"],
                    use_container_width=True, hide_index=True
                )
            else:
                st.caption("—")

        st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)
        if idx_right is not None and st.checkbox("Tampilkan gabungan (unique by employee_id)"):
            left_df = st.session_state.shortlists[idx_left]["df"]
            right_df = st.session_state.shortlists[idx_right]["df"]
            combined = pd.concat([left_df, right_df], ignore_index=True)\
                        .drop_duplicates(subset=["employee_id"])
            st.dataframe(combined, use_container_width=True, hide_index=True)

