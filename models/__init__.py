import joblib
import os

try:
    model_path = os.path.join(os.path.dirname(__file__), 'best_resume_scorer.pkl')
    resume_scorer_model = joblib.load(model_path)  # Ganti nama variabel
except Exception as e:
    raise ImportError(f"Failed to load resume scoring model: {str(e)}")