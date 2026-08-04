# This app intentionally has no ORM models.
#
# Prediction history is persisted to a flat CSV file (see
# predictor/utils.py + settings.PREDICTION_HISTORY_PATH) instead of a
# database table, per the project's storage requirements.
