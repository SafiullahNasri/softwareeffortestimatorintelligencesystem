# EffortAI - Software Effort Intelligence

Flat project: every file lives in the repo root. No folders needed.

Files (all upload to the SAME level on GitHub):
- app.py                 main Streamlit file (set this as "Main file path")
- effort_schema.py       feature ranges, presets, validation
- effort_model.py        model loading, prediction, intervals, Shapley
- effort_analytics.py    Monte Carlo, schedule, staffing, risk
- effort_reporting.py    HTML report export
- effort_theme.py        light/dark green theme
- requirements.txt       dependencies
- model.pkl              YOUR trained model (you add this)

Deploy: Streamlit Community Cloud -> Create app -> pick repo -> Main file path: app.py
