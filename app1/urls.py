from django.urls import path, include
from app1 import views

urlpatterns = [
    path('', views.home, name="index"),
    path('home', views.home, name="home"),
    path('phase1_task1', views.phase1_task1, name="phase1_task1"),
    path('phase1_task1_refresh', views.phase1_task1_refresh, name="phase1_task1_refresh"),
    path('phase1_task2', views.phase1_task2, name="phase1_task2"),
    path('phase1_task2_refresh', views.phase1_task2_refresh, name="phase1_task2_refresh"),
    path('binance_ticker_24hr', views.binance_ticker_24hr, name="binance_ticker_24hr"),
    path('original_netflix_dataset', views.original_netflix_dataset, name="original_netflix_dataset"),
    path('cleaned_netflix_dataset', views.cleaned_netflix_dataset, name="cleaned_netflix_dataset"),
]