# ⚡ DoublingIQ
AI-Powered Energy Prediction & Efficiency Monitoring System for a Textile Doubling Mill.

This project is a free, GitHub-ready starter based on the supplied DoublingIQ specification: ML energy prediction, dashboard, efficiency, machine comparison, history, synthetic/demo data, CSV validation, and English/Tamil UI.

## Run in GitHub Codespaces
Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Frontend (second terminal):
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```
Open the Vite port shown by Codespaces.

The demo uses clearly labeled synthetic data. Replace it with real mill data before operational use.
