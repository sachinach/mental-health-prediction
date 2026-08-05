from django.test import Client, TestCase
from django.urls import reverse

from .utils import load_model, predict_risk


class ModelLoadingTests(TestCase):
    def test_model_and_preprocessing_load(self):
        model, preprocessing = load_model()
        self.assertIsNotNone(model)
        self.assertIn("feature_names", preprocessing)


class PredictionPipelineTests(TestCase):
    def setUp(self):
        self.sample_input = {
            "age": 22, "gender": "Female", "occupation": "Student",
            "sleep_hours": 7.5, "sleep_quality": 7, "social_media_hours": 2.0,
            "academic_work_pressure": 4, "physical_activity_days": 4,
            "stress_level": 3, "anxiety_score": 3, "depression_score": 2,
            "work_life_balance": 7, "mood_score": 7, "concentration_level": 7,
            "social_support": 7,
        }

    def test_predict_risk_returns_expected_keys(self):
        result = predict_risk(self.sample_input)
        for key in ("prediction", "label", "confidence", "probability_high_risk", "recommendations"):
            self.assertIn(key, result)
        self.assertIn(result["label"], ("High Risk", "Low Risk"))


class ViewSmokeTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_page_loads(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)

    def test_predict_page_loads(self):
        response = self.client.get(reverse("predict"))
        self.assertEqual(response.status_code, 200)

    def test_about_page_loads(self):
        response = self.client.get(reverse("about"))
        self.assertEqual(response.status_code, 200)

    def test_history_page_loads(self):
        response = self.client.get(reverse("history"))
        self.assertEqual(response.status_code, 200)

    def test_analytics_page_loads(self):
        response = self.client.get(reverse("analytics"))
        self.assertEqual(response.status_code, 200)
