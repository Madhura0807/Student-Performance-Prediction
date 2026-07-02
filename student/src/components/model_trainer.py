from math import sqrt

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.components.data_transformation import DataTransformation
from src.config import ARTIFACT_DIR, METRICS_PATH, MODEL_PATH
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_json, save_object

logger = get_logger(__name__)


class ModelTrainer:
    """Train and compare multiple regression models for student performance prediction."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.target_column = "Math Score"
        self.feature_columns = [
            "Gender",
            "Race/Ethnicity",
            "Parental Level of Education",
            "Lunch",
            "Test Preparation Course",
            "Reading Score",
            "Writing Score",
        ]

    def train(self):
        try:
            X = self.data[self.feature_columns]
            y = self.data[self.target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            preprocessor = DataTransformation().get_preprocessor()

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1),
                "CatBoost": CatBoostRegressor(iterations=250, depth=6, learning_rate=0.1, loss_function="RMSE", silent=True),
            }

            results = {}
            best_model_name = None
            best_model = None
            best_score = -float("inf")

            for name, model in models.items():
                pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                metrics = self._evaluate_model(y_test, predictions)
                results[name] = metrics

                if metrics["r2_score"] > best_score:
                    best_score = metrics["r2_score"]
                    best_model_name = name
                    best_model = pipeline

            save_json(METRICS_PATH, results)
            save_object(MODEL_PATH, best_model)
            logger.info("Training complete. Best model: %s", best_model_name)

            return {
                "best_model_name": best_model_name,
                "best_score": round(best_score, 4),
                "results": results,
                "model_path": str(MODEL_PATH),
            }
        except Exception as exc:
            raise CustomException(f"Training failed: {exc}") from exc

    def _evaluate_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
            "mse": round(mse, 4),
            "rmse": round(sqrt(mse), 4),
            "r2_score": round(r2_score(y_true, y_pred), 4),
        }
