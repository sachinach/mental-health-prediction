import csv
import logging

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect, render

from .forms import MentalHealthPredictionForm
from .utils import (ModelLoadError, PredictionError, get_model_metrics,
                     predict_risk, read_history, save_prediction)

logger = logging.getLogger("predictor")


def home(request):
    return render(request, "home.html")


def predict(request):
    """Render the prediction form and handle submission."""
    if request.method == "POST":
        form = MentalHealthPredictionForm(request.POST)
        if form.is_valid():
            raw_data = form.cleaned_data
            try:
                result = predict_risk(raw_data)
                save_prediction(raw_data, result)
            except ModelLoadError as exc:
                logger.error("Model load error: %s", exc)
                messages.error(request, "The prediction model is currently unavailable. Please try again shortly.")
                return render(request, "predict.html", {"form": form})
            except PredictionError as exc:
                logger.warning("Prediction error: %s", exc)
                messages.error(request, f"Couldn't generate a prediction: {exc}")
                return render(request, "predict.html", {"form": form})

            request.session["last_result"] = result
            request.session["last_input"] = raw_data
            return redirect("result")
        messages.error(request, "Please correct the highlighted fields and try again.")
    else:
        form = MentalHealthPredictionForm()

    return render(request, "predict.html", {"form": form})


def result(request):
    """Display the most recent prediction stored in the session."""
    result_data = request.session.get("last_result")
    input_data = request.session.get("last_input")

    if not result_data:
        messages.info(request, "Please submit the form first to see a prediction.")
        return redirect("predict")

    return render(request, "result.html", {"result": result_data, "input": input_data})


def history(request):
    """Display prediction history with search support (client-side
    pagination is handled in the template/JS)."""
    records = read_history()

    query = request.GET.get("q", "").strip().lower()
    if query:
        records = [
            r for r in records
            if query in r.get("Gender", "").lower()
            or query in r.get("Occupation", "").lower()
            or query in r.get("Prediction", "").lower()
            or query in r.get("Age", "").lower()
        ]

    return render(request, "history.html", {"records": records, "query": query})


def download_history(request):
    """Serve the raw prediction history CSV as a download."""
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="prediction_history.csv"'

    if settings.PREDICTION_HISTORY_PATH.exists():
        with open(settings.PREDICTION_HISTORY_PATH, "r", encoding="utf-8") as f:
            response.write(f.read())
    else:
        writer = csv.writer(response)
        writer.writerow(["Timestamp", "Age", "Gender", "Occupation", "Prediction", "Probability"])

    return response


def about(request):
    try:
        metrics = get_model_metrics()
    except ModelLoadError:
        metrics = None
    return render(request, "about.html", {"metrics": metrics})


def error_404(request, exception=None):
    return render(request, "404.html", status=404)


def error_500(request):
    return render(request, "500.html", status=500)
