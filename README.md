# Leaderboard Setup

### Frontend Setup

1. Open the HTML file in a web browser:
   ```bash
   open static/index.html
   ```

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Asap7772/leaderboard
   cd leaderboard
   ```

2. Install dependencies:
   ```bash
   conda create -n leaderboard_env python=3.10
   pip install uv
   uv pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 10000 --reload
   ```

### Docker Setup for Google Cloud Run

1. Setup Container and run
```bash
export TAG=initial_build
docker build -t $TAG .
export PORT=10000
docker run -dp $PORT:$PORT -e PORT=$PORT $TAG
```

2. Verify if running
```bash
curl http://127.0.0.1:$PORT
```

3. Setup Google Cloud Run
[Link to Setup](https://github.com/sekR4/FastAPI-on-Google-Cloud-Run)