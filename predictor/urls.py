from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict, name="predict"),
    path("result/", views.result, name="result"),
    path("history/", views.history, name="history"),
    path("history/download/", views.download_history, name="download_history"),
    path("about/", views.about, name="about"),
    path("analytics/", views.analytics, name="analytics"),
]
