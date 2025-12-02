Service + Dashboard scaffold

How to run locally (without docker):

1. Create/activate a Python environment.
2. Install service requirements:
   pip install -r services/service/requirements.txt
   python src/service/server.py

3. Install dashboard requirements and run:
   pip install -r services/dashboard/requirements.txt
   streamlit run src/dashboard/app.py

With docker-compose:
  docker-compose up --build

Place your trained model at `models/model.pkl` (joblib/pickle) before running the service.

If you prefer the service to retrain the model automatically when the model file is missing, start the service with the env var `RETRAIN_IF_MISSING=1` set. Example:

```bash
RETRAIN_IF_MISSING=1 python src/service/server.py
```

Or with docker-compose, add the environment variable under the `service` section if you want the container to retrain on startup.
