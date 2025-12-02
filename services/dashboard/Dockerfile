FROM python:3.10-slim
WORKDIR /app

COPY services/dashboard/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY atp_tennis.csv /app/atp_tennis.csv
COPY models /app/models
COPY src /app/src

WORKDIR /app
ENV PYTHONPATH=/app

EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
