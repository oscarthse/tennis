from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/model.pkl')

app = Flask(__name__)


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


model = load_model()

# If model not found and the env var RETRAIN_IF_MISSING is set, attempt to train and export a model
if model is None and os.environ.get('RETRAIN_IF_MISSING') == '1':
    try:
        # import the train script from repo root
        import sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from train_and_export import train_and_export
        print('Model not found. Retraining because RETRAIN_IF_MISSING=1...')
        train_and_export(model_path=MODEL_PATH)
        model = load_model()
        print('Retrain complete, model loaded.')
    except Exception as e:
        print('Retrain failed:', e)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'model not found on server'}), 500

    payload = request.get_json()
    if payload is None:
        return jsonify({'error': 'invalid json payload'}), 400

    # Expect features as a list or 2d-list; try to normalize
    X = payload.get('X')
    if X is None:
        return jsonify({'error': 'payload must contain key "X" with feature vector(s)'}), 400

    try:
        # Convert to DataFrame with proper column names
        import pandas as pd
        columns = ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Best of', 'Series', 'Court', 'Surface', 'Round']
        X_df = pd.DataFrame(X, columns=columns)
        preds = model.predict(X_df).tolist()
    except Exception as e:
        return jsonify({'error': f'prediction failed: {e}'}), 500

    return jsonify({'predictions': preds})


if __name__ == '__main__':
    # Allow overriding the port with the PORT env var (useful when 5000 is in use)
    try:
        port = int(os.environ.get('PORT', '5000'))
    except Exception:
        port = 5000
    app.run(host='0.0.0.0', port=port)
