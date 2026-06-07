# Deployment notes

The project is set up as a local API demo rather than a cloud production service.

## Local Python run

Use Python 3.10 and the pinned dependencies:

```cmd
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip install -e .
python -m uvicorn --app-dir src artrec.api.main:app --host 127.0.0.1 --port 8000
```

## Docker run

```cmd
docker build -t artrec-platform .
docker run --rm -p 8000:8000 artrec-platform
```

For the API plus Prometheus demo:

```cmd
docker compose up --build
```

## What is missing for production

Before a real deployment, I would add managed secrets, TLS, private metrics access, request logging, rate limiting, CI/CD promotion gates, container scanning, rollback, and a staging environment.

The current setup is still useful for a recruiter or reviewer because they can run the API, hit the endpoints, inspect metrics, and see the project structure without needing cloud credentials.
