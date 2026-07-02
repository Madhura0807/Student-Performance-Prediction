from flask import Flask, jsonify, render_template, request

from src.config import ARTIFACT_DIR, MODEL_PATH, STATIC_DIR, TEMPLATES_DIR
from src.exception import CustomException
from src.logger import get_logger
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import load_json

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
logger = get_logger(__name__)


def ensure_model_ready():
    if MODEL_PATH.exists():
        return None

    logger.info("Model artifact missing. Training a new model automatically.")
    pipeline = TrainPipeline()
    return pipeline.run()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        ensure_model_ready()
        return render_template("predict.html")

    try:
        ensure_model_ready()
        gender = request.form.get("gender", "").strip()
        race = request.form.get("race", "").strip()
        education = request.form.get("education", "").strip()
        lunch = request.form.get("lunch", "").strip()
        prep = request.form.get("preparation", "").strip()
        reading_score_raw = request.form.get("reading_score", "")
        writing_score_raw = request.form.get("writing_score", "")

        if not all([gender, race, education, lunch, prep, reading_score_raw, writing_score_raw]):
            raise CustomException("Please fill in all required fields.")

        reading_score = float(reading_score_raw)
        writing_score = float(writing_score_raw)

        form_data = {
            "Gender": gender,
            "Race/Ethnicity": race,
            "Parental Level of Education": education,
            "Lunch": lunch,
            "Test Preparation Course": prep,
            "Reading Score": reading_score,
            "Writing Score": writing_score,
        }

        if not (0 <= reading_score <= 100 and 0 <= writing_score <= 100):
            raise CustomException("Reading and Writing scores must be between 0 and 100.")

        pipeline = PredictPipeline()
        prediction = pipeline.predict(form_data)
        interpretation = interpret_prediction(prediction)
        return render_template("result.html", prediction=prediction, interpretation=interpretation, form_data=form_data)
    except Exception as exc:
        logger.exception("Prediction request failed")
        return render_template("error.html", error=str(exc))


@app.route("/model")
def model_info():
    ensure_model_ready()
    metrics_path = ARTIFACT_DIR / "model_metrics.json"
    visuals = {}
    if metrics_path.exists():
        try:
            visuals = load_json(ARTIFACT_DIR / "visualizations" / "summary.json")
        except FileNotFoundError:
            visuals = {}
    return render_template("model.html", visuals=visuals)


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/train")
def train_model():
    try:
        pipeline = TrainPipeline()
        summary = pipeline.run()
        return jsonify({"status": "success", "summary": summary})
    except Exception as exc:
        logger.exception("Training failed")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.errorhandler(404)
def page_not_found(error):
    return render_template("error.html", error="The requested page could not be found."), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error="The server encountered an internal error."), 500


def interpret_prediction(score: float) -> str:
    if score >= 85:
        return "Excellent performance is expected. The student is likely to excel academically."
    if score >= 70:
        return "Good performance is expected. The student is likely to perform well overall."
    if score >= 55:
        return "Average performance is expected. The student may need extra guidance in some areas."
    return "Performance is below expectations. Additional academic support is recommended."


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
