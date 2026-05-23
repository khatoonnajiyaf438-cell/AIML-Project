from flask import Flask, render_template, request
import pickle
from pathlib import Path
from utils import run_checks

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = BASE_DIR.parent / "model"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

with open(MODEL_DIR / "model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open(MODEL_DIR / "vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


def default_form_data() -> dict[str, str]:
    return {
        "job": "",
        "company": "",
        "check_type": "Full Check",
        "check_value": "",
    }


@app.route("/")
def home():
    return render_template("index.html", form_data=default_form_data())


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {
        "job": request.form.get("job", "").strip(),
        "company": request.form.get("company", "").strip(),
        "check_type": request.form.get("check_type", "Full Check").strip() or "Full Check",
        "check_value": request.form.get("check_value", "").strip(),
    }

    if not form_data["job"] or not form_data["company"] or not form_data["check_value"]:
        prediction = {
            "message": "Please fill job description, company name, and the selected check value.",
            "verdict": "Incomplete",
            "flags": ["All required fields must be filled before analysis."],
        }
        return render_template("index.html", prediction=prediction, form_data=form_data)

    vec = vectorizer.transform([form_data["job"]])
    ml_result = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100

    rule_result = run_checks(
        job_text=form_data["job"],
        company=form_data["company"],
        check_type=form_data["check_type"],
        check_value=form_data["check_value"],
    )

    is_fake = bool(ml_result == 1 or rule_result["is_fake"])
    if is_fake:
        verdict = "Fake Job"
        message = f"Fake Job ❌ (Confidence: {confidence:.2f}%)"
    else:
        verdict = "Real Job"
        message = f"Real Job ✅ (Confidence: {confidence:.2f}%)"

    prediction = {
        "message": message,
        "verdict": verdict,
        "flags": rule_result["flags"],
        "detected_type": rule_result["detected_type"],
    }
    return render_template("index.html", prediction=prediction, form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)
