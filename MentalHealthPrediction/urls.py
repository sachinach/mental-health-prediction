from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("predictor.urls")),
]

handler404 = "predictor.views.error_404"
handler500 = "predictor.views.error_500"
