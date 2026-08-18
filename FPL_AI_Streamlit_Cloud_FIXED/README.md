# FPL AI 2026/27 — Streamlit Cloud FIXED

Tato verze opravuje chybu z předchozího deploymentu.

## Oprava
Původní `requirements.txt` explicitně vyžadoval `pyarrow>=16,<22`. Streamlit Cloud se pokusil pro Python 3.14 sestavit PyArrow ze zdrojů a skončil na `cmake: No such file or directory`. Tato verze PyArrow explicitně nevyžaduje a lokální snapshoty ukládá jako CSV.

## Deployment
- GitHub repository: nahraj obsah této složky do kořene repozitáře.
- Streamlit entrypoint: `run.py`
- V Advanced settings zvol Python 3.12.
- Deploy.

Streamlit Community Cloud standardně používá Python 3.12; Python verzi je možné vybrat v Advanced settings při deploymentu.
