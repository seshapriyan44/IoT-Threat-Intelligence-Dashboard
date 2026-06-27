from predict import predict_attack

from flask import (
    Flask,
    render_template,
    request
)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    selected_model = request.form.get(
        "model"
    )

    uploaded_file = request.files.get(
        "csvfile"
    )

    if uploaded_file is None:

        return render_template(
            "result.html",
            status="No file uploaded",
            model="N/A",
            error="N/A",
            threshold="N/A",
            severity="N/A"
        )

    save_path = (
        "uploads/"
        + uploaded_file.filename
    )

    uploaded_file.save(
        save_path
    )

    result = predict_attack(
        save_path,
        selected_model
    )

    severity = (
        "HIGH"
        if result["status"] == "ATTACK"
        else "LOW"
    )

    # Select confusion matrix based on chosen model
    if selected_model == "Federated":
        confusion_matrix = "charts/federated_confusion.png"
    else:
        confusion_matrix = "charts/centralized_confusion.png"
        
    if result["error_ratio"] > 100:
        ratio_display = ">100"
    else:
        ratio_display = f'{result["error_ratio"]:.3f}'
    
    uploaded_filename = uploaded_file.filename

    return render_template(

        "result.html",

        status=result["status"],
        model=selected_model,
        error=result["error"],
        threshold=result["threshold"],
        error_ratio=result["error_ratio"],
        severity=severity,
        
        ratio_display=ratio_display,
        
        threat_score=result["threat_score"],

        top_features=result["top_features"],
        feature_chart=result["feature_chart"],
        confusion_matrix=confusion_matrix,
        
        uploaded_filename=uploaded_filename

    )


if __name__ == "__main__":
    app.run(
        debug=True
    )