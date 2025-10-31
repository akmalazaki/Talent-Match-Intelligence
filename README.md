<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/PostgreSQL-DB-336791?style=for-the-badge&logo=postgresql" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/OpenRouter-AI-6E56CF?style=for-the-badge&logo=openai" alt="OpenRouter">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🧠 Talent Match Intelligence Dashboard</h1>

<p align="center">
  <i>AI-augmented HR analytics system for talent readiness and succession planning.</i><br>
  <i>Built with Streamlit · PostgreSQL · OpenRouter AI · Plotly</i>
</p>

---

<p align="center">
  🚀 <b>Live Dashboard:</b><br>
  <a href="https://talent-match-intelligence-dashboard-by-akmal.streamlit.app/" target="_blank">
    https://talent-match-intelligence-dashboard-by-akmal.streamlit.app/
  </a>
</p>

---

### 🎯 Overview

**Talent Match Intelligence** is an AI-powered HR analytics dashboard designed to help leaders identify what makes top performers successful and uncover employees who share similar potential.  
The system combines **data analytics**, **SQL logic**, and **LLM-generated insights** to support **succession planning** and **leadership development**.

---

### 🧩 Key Features

- 🧭 **Vacancy Definition** – Define benchmark roles, job levels, and purpose for succession planning.  
- 🤖 **AI Job Profile Generation** – Automatically generate job requirements, key competencies, and responsibilities via OpenRouter AI.  
- 📈 **Talent Readiness Analytics** – Measure competency alignment through execution, strategic, and leadership dimensions.  
- 🧭 **AI Advisory Narrative** – Generate personalized HR advisory reports with readiness assessment and development recommendations.  
- 📄 **One-Click PDF Reports** – Export candidate insights and AI summaries to professional HR reports.  
- 🗂 **Shortlist Comparison** – Save, compare, and analyze shortlisted successors across sessions.  

---

### 🧠 Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | Streamlit, Plotly |
| Backend | PostgreSQL, SQLAlchemy |
| AI Engine | OpenRouter (Mistral-7B) |
| Language | Python 3.11 |
| Reporting | FPDF |
| Deployment | Streamlit App (local / cloud) |

---

### 🧭 Core Concept
> “Turning performance data into succession insight — Talent Match Intelligence helps leaders see not just who performs today, but who will lead tomorrow.”

---

## 🧩 Project Objectives

1. **Discover the Pattern of Success**  
   Analyze competencies, psychometric profiles, behavioral strengths, and contextual factors to understand *why* certain employees achieve top performance.  

2. **Operationalize the Logic in SQL**  
   Convert analytical findings into SQL transformations to quantify success drivers and generate the **Final Match Rate** per employee.

3. **Build the Talent Intelligence Dashboard**  
   Visualize readiness, generate AI-based job profiles, and provide actionable HR insights through Streamlit and OpenRouter integration.

---

## 🧭 Step 1 — Discover the Pattern of Success

Explored key success factors from:
- **Competency Pillars** (`competencies_yearly` + `dim_competency_pillars`)
- **Psychometric Profiles** (`profiles_psych`, `papi_scores`)
- **Behavioral Strengths** (`strengths`)
- **Contextual Data** (`grade`, `education`, `years_of_service_months`)

Visualizations included:
- Heatmaps & boxplots for distribution analysis  
- Correlation plots (High Performers only)  
- Comparison matrices between success groups  
- Effect size (Cohen’s d) analysis for variable influence  

> **Insight:** High performers show consistently higher competency alignment and slightly elevated psychometric indices, with Execution and Strategic Value as the most differentiating pillars.

---

## 🧮 Step 2 — Operationalize the Logic in SQL

Translated findings into SQL transformations:
- Calculated **average pillar score per employee**
- Computed **ratio vs. benchmark** (based on high performers)
- Derived **Execution, Strategic, and Leadership Match Rates**
- Combined into a unified **Final Success Score**

Example snippet:
sql
`final_match_rate = (exec_match_rate + strat_match_rate + lead_match_rate) / 3`

---

### 📊 Defined Readiness Levels

| Category | Success Score | Meaning |
|-----------|---------------|----------|
| **Strong Successor** | ≥ 105 | Ready for immediate succession |
| **Ready Soon** | ≥ 95 | Ready within 6–12 months |
| **Development Priority** | < 80 | Requires further development |

SQL logic was fully validated in **PostgreSQL** and later integrated into the **Streamlit dashboard** for real-time analysis and visualization.

---

## 📊 Step 3 — Talent Intelligence Dashboard

Built using **Streamlit**, **Plotly**, and **OpenRouter AI**, the dashboard provides both **quantitative analytics** and **AI-assisted HR insights** for succession planning and leadership readiness.

---

### 🧭 Vacancy Definition — Role Context

Leaders define:
- **Role Name**  
- **Job Level**  
- **Role Purpose**  
- **Benchmark Employees (High Performers)**  

This step personalizes the analysis by aligning the benchmark group with the specific role and level under review.

---

### 🤖 AI-Generated Job Profile

Powered by **OpenRouter (Mistral-7B)**, the system automatically generates:
- Job descriptions  
- Key requirements  
- Core competencies  

These AI-generated profiles help align competency expectations with real organizational context.

---

### 📈 Talent Match Summary

Three **readiness-level cards** summarize the overall bench strength within the selected role:
- **Strong Successor (≥105%)**
- **Ready Soon (≥95%)**
- **Development Priority (<80%)**

Each category represents how close employees are to the success profile of top performers.

---

### 🏅 Candidate Ranking — Benchmark Fit

Displays a ranked list of all employees based on:
- `final_match_rate`
- `readiness` label
- Development classification  

This table is fully **exportable as CSV** and serves as the foundation for HR discussions during succession meetings.

---

### 🔬 Candidate Deep Dive — Strength Gap Analysis

Provides a detailed **Talent Growth Vector (TGV)** comparison between each candidate and the benchmark profile.  
Includes:
- **Radar charts** (competency alignment)  
- **Bar charts** (strength gap vs benchmark)  
- **Individual TGV breakdowns** across Execution, Strategic, and Leadership domains  

---

### 🎯 HR Advisory Narrative

An **AI-generated HR advisory narrative** summarizes each candidate’s readiness in three concise parts:
1. **Readiness assessment** — overall performance and match reasoning.  
2. **Key strengths** — standout competencies contributing to readiness.  
3. **Development recommendations** — targeted action plan for the next 6–12 months.  

This narrative bridges **quantitative data with qualitative judgment**, helping HR leaders move from data interpretation to **actionable development decisions**.

---

> *In essence, Step 3 transforms analytical outputs into an interactive, AI-driven decision support system — empowering HR teams to visualize, interpret, and act upon succession insights in real time.*



<p align="center">
  <img src="screenshots/preview_dashboard.png" width="85%">
  <br>
  <i>Example view of Talent Match Intelligence Dashboard</i>
</p>
