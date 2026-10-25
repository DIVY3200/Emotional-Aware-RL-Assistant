# Web demo image (simulator playground + photo tab). The live webcam assistant
# needs local camera, audio and a display, so run that one outside Docker.
#
#   docker build -t emotion-rl .
#   docker run -p 8501:8501 -e GEMINI_API_KEY=... emotion-rl
#   open http://localhost:8501
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[web,vision]" google-genai python-dotenv

COPY configs ./configs
COPY models ./models

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/emotion_rl/web_demo.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
