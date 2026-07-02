from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
NOTEBOOK_DIR = ROOT_DIR / "notebook"
SRC_DIR = ROOT_DIR / "src"
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
LOG_DIR = ROOT_DIR / "logs"
PLOT_DIR = STATIC_DIR / "plots"
DATA_PATH = ARTIFACT_DIR / "student_performance.csv"
MODEL_PATH = ARTIFACT_DIR / "student_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
VISUALIZATION_DIR = ARTIFACT_DIR / "visualizations"
CATBOOST_INFO_DIR = ROOT_DIR / "catboost_info"
