from django.urls import path, include
from app1 import views

urlpatterns = [
    path('', views.home, name="index"),
    path('home', views.home, name="home"),

    path('phase1_task1', views.phase1_task1, name="phase1_task1"),
    path('binance_ticker_24hr', views.binance_ticker_24hr, name="binance_ticker_24hr"),
    path('phase1_task1_refresh', views.phase1_task1_refresh, name="phase1_task1_refresh"),
    
    path('original_netflix_dataset', views.original_netflix_dataset, name="original_netflix_dataset"),
    
    path('phase1_task2', views.phase1_task2, name="phase1_task2"),
    path('phase1_task2_refresh', views.phase1_task2_refresh, name="phase1_task2_refresh"),
    path('phase1_task1_cleaned_dataset', views.phase1_task1_cleaned_dataset, name="phase1_task1_cleaned_dataset"),

    path('phase2_task1', views.phase2_task1, name="phase2_task1"),
    path('phase2_task1_refresh', views.phase2_task1_refresh, name="phase2_task1_refresh"),
    path('phase2_task1_cleaned_dataset', views.phase2_task1_cleaned_dataset, name="phase2_task1_cleaned_dataset"),

    path('phase2_task2', views.phase2_task2, name="phase2_task2"),
    path('phase2_task2_model_refresh', views.phase2_task2_model_refresh, name="phase2_task2_model_refresh"),
    path('phase2_task2_train_dataset', views.phase2_task2_train_dataset, name="phase2_task2_train_dataset"),
    path('phase2_task2_test_dataset', views.phase2_task2_test_dataset, name="phase2_task2_test_dataset"),
    path('phase2_task2_prediction', views.phase2_task2_prediction, name="phase2_task2_prediction"),
    path('phase2_task2_model_evaluation', views.phase2_task2_model_evaluation, name="phase2_task2_model_evaluation"),
]