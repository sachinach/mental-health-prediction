# Mental Health Risk Prediction

A production-style Django web application that screens for mental health
risk (Low Risk / High Risk) from a short lifestyle and wellbeing survey,
using a Random Forest classifier trained on behavioural survey data.

> Not a diagnostic tool. This project is a portfolio-grade demonstration
> of shipping a trained ML model behind a real web application, not a
> substitute for professional clinical assessment.

---

## Overview

The model takes thirteen inputs — sleep, stress, workload, social
support and related signals — and returns a risk label with a
confidence score, a risk meter, and a short list of suggested next
steps. Every prediction is appended to a CSV-backed history log that's
searchable, paginated, and downloadable from the app.

## Features

- **Home** — hero section, methodology teaser, feature highlights.
- **Prediction** — a validated Bootstrap form covering every model
  feature, with client-side validation and a loading state.
- **Result** — risk label, confidence, animated risk meter, and
  personalised recommendations.
- **History** — searchable, paginated table of past predictions with a
  one-click CSV export.
- **About** — business problem, objective, ML workflow, algorithms
  compared, and model metrics.
- Dark mode toggle, toast notifications, custom 404/500 pages, and a
  fully responsive layout.

## Tech stack

| Layer      | Choice                                   |
|------------|-------------------------------------------|
| Backend    | Python, Django, Pandas, NumPy, scikit-learn |
| Frontend   | HTML5, CSS3, Bootstrap 5, vanilla JS       |
| ML storage | Pickle (`model.pkl`, `preprocessing.pkl`)  |
| Data store | Flat CSV (`dataset/prediction_history.csv`) — no SQL database for prediction data |

## Model summary

- **Target:** binary risk label derived from a clinician-style severity
  rating (Mild → Low Risk; Moderate/Severe → High Risk).
- **Pipeline:** cleaning → feature engineering (Wellbeing Index,
  Pressure Index) → encoding (label + one-hot) → correlation-based
  feature selection → standard scaling → SMOTE-style class balancing on
  the training split → comparison across 7 classifiers → grid-search
  tuning → final model selection.
- **Final model:** Random Forest — see the About page in the running app
  for the exact held-out test metrics.
- Full methodology and code lives in `train_pipeline.py` at the project
  root, which regenerates `model/model.pkl` and `model/preprocessing.pkl`
  from `dataset/mental_health_prediction.csv`.

## Folder structure

```
MentalHealthPrediction/
├── manage.py
├── requirements.txt
├── README.md
├── .gitignore
├── Procfile
├── runtime.txt
├── train_pipeline.py
├── model/
│   ├── model.pkl
│   └── preprocessing.pkl
├── dataset/
│   ├── mental_health_prediction.csv
│   └── prediction_history.csv
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── images/
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── predict.html
│   ├── result.html
│   ├── history.html
│   ├── about.html
│   ├── 404.html
│   └── 500.html
├── predictor/
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   ├── utils.py
│   ├── models.py
│   ├── admin.py
│   ├── apps.py
│   └── tests.py
└── MentalHealthPrediction/
    ├── settings.py
    ├── urls.py
    ├── wsgi.py
    └── asgi.py
```

## Installation

```bash
git clone https://github.com/<your-username>/mental-health-risk-prediction.git
cd mental-health-risk-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Run locally

```bash
python manage.py migrate
python manage.py runserver
```

Visit `http://127.0.0.1:8000`.

To regenerate the model artifacts from the dataset:

```bash
python train_pipeline.py
```

This writes fresh `model.pkl` and `preprocessing.pkl` files into
`model/`.

## Screenshots

_Add screenshots here once deployed, e.g.:_

```
docs/screenshots/home.png
docs/screenshots/predict.png
docs/screenshots/result.png
docs/screenshots/history.png
```

## Deployment

### Render

1. Push this repo to GitHub.
2. Create a new **Web Service** on Render, pointing at the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn MentalHealthPrediction.wsgi:application`
5. Set environment variables: `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=False`,
   `DJANGO_ALLOWED_HOSTS=<your-service>.onrender.com`.

### Railway

1. `railway init` inside the project folder.
2. Railway auto-detects `Procfile` and `runtime.txt`.
3. Set the same environment variables as above via `railway variables set`.
4. `railway up`.

### PythonAnywhere

1. Upload the repo (or `git clone` from the Bash console).
2. Create a virtualenv and `pip install -r requirements.txt`.
3. Point the WSGI configuration file at
   `MentalHealthPrediction.wsgi.application`.
4. Set `DJANGO_DEBUG=False` and your domain in
   `DJANGO_ALLOWED_HOSTS` via the PythonAnywhere environment variables
   panel, then reload the web app.

### Environment variables

| Variable                | Purpose                                 | Default          |
|--------------------------|------------------------------------------|-------------------|
| `DJANGO_SECRET_KEY`      | Django secret key                        | insecure dev key |
| `DJANGO_DEBUG`           | `True` / `False`                         | `True`           |
| `DJANGO_ALLOWED_HOSTS`   | Comma-separated allowed hosts            | `127.0.0.1,localhost` |

## Future scope

- Add SHAP-based explainability to the result page.
- Move prediction history to a proper time-series store for
  longitudinal tracking per user.
- Add authenticated accounts so history is private per user.
- Periodic retraining pipeline with a versioned model registry.

## License

MIT — free to use for learning and portfolio purposes.
