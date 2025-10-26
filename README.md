# Cell Viability Analyzer - Real-Time Web Application

A real-time web application for analyzing cell viability from microscope images.

## Features
- Real-time synchronization across devices
- Camera capture & image upload
- Automatic live/dead cell detection
- Multi-device support

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
Open `frontend/index.html` in your browser.

## Deployment
See DEPLOYMENT.md for full deployment instructions.

## API Endpoints
- `GET /health` - Health check
- `POST /analyze` - Analyze single image
- `POST /batch-analyze` - Analyze multiple images

## Tech Stack
- Backend: Flask, OpenCV, scikit-image
- Frontend: React, Tailwind CSS
- Deployment: Render.com + GitHub Pages
