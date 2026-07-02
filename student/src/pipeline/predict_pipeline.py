import pandas as pd

from src.config import MODEL_PATH
from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_object

logger = get_logger(__name__)


class PredictPipeline:
    """Load the trained model and predict math score for a new student profile."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = load_object(self.model_path)
            logger.info("Model loaded successfully.")
        except Exception as exc:
            raise CustomException(f"Unable to load model: {exc}") from exc

    def predict(self, features: dict) -> float:
        try:
            input_frame = pd.DataFrame([features])
            prediction = self.model.predict(input_frame)[0]
            return round(float(prediction), 2)
        except Exception as exc:
            raise CustomException(f"Prediction failed: {exc}") from exc
