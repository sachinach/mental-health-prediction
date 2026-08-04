from django import forms

from .utils import get_form_choices


class MentalHealthPredictionForm(forms.Form):
    """Collects the exact set of inputs the trained model was fit on.
    Choice fields are populated dynamically from preprocessing.pkl so the
    form can never drift out of sync with what the model expects.
    """

    age = forms.IntegerField(
        min_value=13, max_value=80,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 22"}),
        help_text="Age in years (13-80)",
    )
    gender = forms.ChoiceField(
        choices=(), widget=forms.Select(attrs={"class": "form-select"}),
    )
    occupation = forms.ChoiceField(
        choices=(), widget=forms.Select(attrs={"class": "form-select"}),
    )
    sleep_hours = forms.FloatField(
        min_value=0, max_value=14,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1", "placeholder": "e.g. 6.5"}),
        help_text="Average hours of sleep per night",
    )
    sleep_quality = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Self-rated sleep quality (1 = poor, 10 = excellent)",
    )
    social_media_hours = forms.FloatField(
        min_value=0, max_value=16,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
        help_text="Average daily social media / screen use, in hours",
    )
    academic_work_pressure = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Perceived academic / work pressure (1 = low, 10 = high)",
    )
    physical_activity_days = forms.FloatField(
        min_value=0, max_value=7,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Days per week with physical activity",
    )
    stress_level = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Overall stress level (1 = low, 10 = high)",
    )
    anxiety_score = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Self-rated anxiety (1 = low, 10 = high)",
    )
    depression_score = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Self-rated depressive symptoms (1 = low, 10 = high)",
    )
    work_life_balance = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Work-life balance (1 = poor, 10 = excellent)",
    )
    mood_score = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Overall mood (1 = low, 10 = high)",
    )
    concentration_level = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Ability to concentrate (1 = poor, 10 = excellent)",
    )
    social_support = forms.FloatField(
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "1"}),
        help_text="Perceived social support (1 = low, 10 = high)",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        choices = get_form_choices()
        self.fields["gender"].choices = [(g, g) for g in choices["genders"]]
        self.fields["occupation"].choices = [(o, o) for o in choices["occupations"]]
