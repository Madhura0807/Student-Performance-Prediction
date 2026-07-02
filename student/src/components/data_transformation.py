import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


class DataTransformation:
    """Create preprocessing pipelines for categorical and numerical features."""

    def __init__(self) -> None:
        self.categorical_features = [
            "Gender",
            "Race/Ethnicity",
            "Parental Level of Education",
            "Lunch",
            "Test Preparation Course",
        ]
        self.numerical_features = ["Reading Score", "Writing Score"]

    def get_preprocessor(self):
        try:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_pipeline, self.categorical_features),
                    ("num", numerical_pipeline, self.numerical_features),
                ]
            )
            logger.info("Preprocessor created successfully.")
            return preprocessor
        except Exception as exc:
            raise CustomException(f"Unable to create preprocessor: {exc}") from exc
