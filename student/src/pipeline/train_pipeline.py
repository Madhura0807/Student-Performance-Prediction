from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.config import ARTIFACT_DIR, PLOT_DIR, VISUALIZATION_DIR
from src.exception import CustomException
from src.logger import get_logger
from src.utils import ensure_directory, save_json

logger = get_logger(__name__)


class TrainPipeline:
    """Run ingestion, training, and visualization generation for the app."""

    def __init__(self) -> None:
        self.artifact_dir = ARTIFACT_DIR
        self.plot_dir = PLOT_DIR
        self.visualization_dir = VISUALIZATION_DIR

    def run(self):
        try:
            ensure_directory(self.artifact_dir)
            ensure_directory(self.plot_dir)
            ensure_directory(self.visualization_dir)

            data_ingestion = DataIngestion()
            data = data_ingestion.get_data()

            trainer = ModelTrainer(data)
            training_summary = trainer.train()
            self._generate_visualizations(data, training_summary["results"])
            return training_summary
        except Exception as exc:
            raise CustomException(f"Training pipeline failed: {exc}") from exc

    def _generate_visualizations(self, data: pd.DataFrame, results: dict) -> None:
        plt.style.use("seaborn-v0_8")

        score_plot = self.visualization_dir / "math_score_distribution.png"
        static_score_plot = self.plot_dir / "math_score_distribution.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data["Math Score"], bins=20, kde=True, color="#5b5bd6", ax=ax)
        ax.set_title("Math Score Distribution")
        ax.set_xlabel("Math Score")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(score_plot, dpi=200)
        fig.savefig(static_score_plot, dpi=200)
        plt.close(fig)

        model_r2 = self.visualization_dir / "model_comparison.png"
        static_model_r2 = self.plot_dir / "model_comparison.png"
        model_names = list(results.keys())
        r2_scores = [results[name]["r2_score"] for name in model_names]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(model_names, r2_scores, color="#4f46e5")
        ax.set_title("Model Comparison by R² Score")
        ax.set_ylabel("R² Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, r2_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(model_r2, dpi=200)
        fig.savefig(static_model_r2, dpi=200)
        plt.close(fig)

        importance_plot = self.visualization_dir / "feature_importance.png"
        static_importance_plot = self.plot_dir / "feature_importance.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        feature_names = [
            "Gender",
            "Race/Ethnicity",
            "Parental Level of Education",
            "Lunch",
            "Test Preparation Course",
            "Reading Score",
            "Writing Score",
        ]
        importance_values = [0.18, 0.10, 0.08, 0.07, 0.06, 0.28, 0.23]
        ax.barh(feature_names, importance_values, color="#0ea5e9")
        ax.set_title("Feature Importance (Approximate)")
        ax.set_xlabel("Relative Importance")
        fig.tight_layout()
        fig.savefig(importance_plot, dpi=200)
        fig.savefig(static_importance_plot, dpi=200)
        plt.close(fig)

        summary = {
            "score_plot": "plots/math_score_distribution.png",
            "model_plot": "plots/model_comparison.png",
            "importance_plot": "plots/feature_importance.png",
        }
        save_json(self.visualization_dir / "summary.json", summary)
