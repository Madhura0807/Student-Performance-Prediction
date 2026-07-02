import numpy as np
import pandas as pd

from src.config import DATA_PATH
from src.exception import CustomException
from src.logger import get_logger
from src.utils import ensure_directory

logger = get_logger(__name__)


class DataIngestion:
    """Load or create a student performance dataset for training."""

    def __init__(self, data_path: str | None = None) -> None:
        self.data_path = data_path or DATA_PATH

    def get_data(self) -> pd.DataFrame:
        try:
            ensure_directory(self.data_path.parent)
            if not self.data_path.exists():
                logger.info("Dataset not found. Creating a synthetic student performance dataset.")
                return self._create_sample_dataset()

            data = pd.read_csv(self.data_path)
            logger.info("Dataset loaded successfully.")
            return data
        except Exception as exc:
            raise CustomException(f"Unable to read dataset: {exc}") from exc

    def _create_sample_dataset(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n_rows = 700

        gender = rng.choice(["Female", "Male"], size=n_rows, p=[0.52, 0.48])
        race = rng.choice(["group A", "group B", "group C", "group D", "group E"], size=n_rows, p=[0.15, 0.25, 0.3, 0.2, 0.1])
        education = rng.choice(["bachelor's degree", "master's degree", "associate's degree", "high school", "some college"], size=n_rows, p=[0.2, 0.15, 0.25, 0.2, 0.2])
        lunch = rng.choice(["standard", "free/reduced"], size=n_rows, p=[0.75, 0.25])
        prep = rng.choice(["completed", "none"], size=n_rows, p=[0.4, 0.6])

        reading = rng.integers(40, 100, size=n_rows)
        writing = rng.integers(35, 100, size=n_rows)

        math = (
            0.4 * reading
            + 0.4 * writing
            + rng.integers(-10, 11, size=n_rows)
            + np.where(lunch == "free/reduced", -8, 2)
            + np.where(prep == "completed", 6, 0)
            + np.where(gender == "Female", 2, 0)
        )
        math = np.clip(math, 0, 100)

        data = pd.DataFrame(
            {
                "Gender": gender,
                "Race/Ethnicity": race,
                "Parental Level of Education": education,
                "Lunch": lunch,
                "Test Preparation Course": prep,
                "Reading Score": reading,
                "Writing Score": writing,
                "Math Score": np.round(math).astype(int),
            }
        )
        data.to_csv(self.data_path, index=False)
        return data
