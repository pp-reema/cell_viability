# Deployment Guide

## Option 1: Render + GitHub Pages (Recommended)

### Backend (Render.com)
1. Create account at render.com
2. New Web Service → Connect GitHub repo
3. Configure:
   - Build: `pip install -r backend/requirements.txt`
   - Start: `cd backend && gunicorn app:app`
4. Copy your backend URL

### Frontend (GitHub Pages)
1. Update `API_URL` in `frontend/index.html` with your Render URL
2. Push to GitHub
3. Enable GitHub Pages in repo settings
4. Access at: `https://username.github.io/repo-name/frontend/`

## Option 2: Local Testing
```bash
# Backend
cd backend
python app.py

# Frontend
Open frontend/index.html in browser
```

## Troubleshooting
- Backend takes 30-50s to wake up on free tier
- Ensure CORS is enabled in app.py
- Check browser console for errors
