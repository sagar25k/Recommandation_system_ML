from django.urls import path
from .views import recommend_products

urlpatterns = [
    path('', recommend_products, name='input_form'),
    path('analyze/', recommend_products, name='analyze_data'),
]
