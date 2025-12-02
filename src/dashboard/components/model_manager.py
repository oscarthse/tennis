import json
import joblib
import streamlit as st
from src.dashboard.config import MODELS_DIR

def discover_models():
    """
    Discover all available models in the models directory.
    Returns:
        dict: {model_name: {'path': Path, 'metadata': dict}}
    """
    models = {}
    if not MODELS_DIR.exists():
        return models

    for pkl_file in MODELS_DIR.glob('*.pkl'):
        meta_file = pkl_file.with_suffix('.json')
        meta = {}
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"Error loading metadata for {pkl_file}: {e}")

        models[pkl_file.stem] = {'path': pkl_file, 'metadata': meta}

    return models

@st.cache_resource
def load_model(model_path):
    """Load a trained model from disk."""
    return joblib.load(model_path)

def get_model_features(model):
    """
    Attempt to extract feature names from the model pipeline.
    Returns:
        list: List of feature names or None if not found.
    """
    try:
        # Assuming sklearn Pipeline
        if hasattr(model, 'named_steps'):
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor:
                # This is tricky with ColumnTransformer, but let's try to get input features
                if hasattr(preprocessor, 'get_feature_names_out'):
                    return list(preprocessor.get_feature_names_out())
    except Exception:
        pass
    return None
