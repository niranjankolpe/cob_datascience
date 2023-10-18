from django.shortcuts import render, redirect

import pandas as pd
import requests

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Create your views here.
def home(request):
    return render(request, "index.html")

def phase1_task1(request):
    return render(request, "phase1_task1.html")

def phase1_task1_refresh(request):
    api_url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(api_url)
    if response.status_code == 200:
        data_json = response.json()
    else:
        print("Failed to load page!\n")
        return redirect("home")
    df = pd.DataFrame(data_json)
    df.to_csv("static/binance_ticker_24hr.csv")
    df.to_html("templates/binance_ticker_24hr.html", table_id="binance_ticker_24hr")
    del api_url, response, data_json, df
    return redirect("phase1_task1")

def phase1_task2(request):
    return render(request, "phase1_task2.html")

def phase1_task2_refresh(request):
    netflix_df = pd.read_csv("static/dataset - netflix1.csv")
    netflix_df.to_html("templates/original_netflix_dataset.html", table_id="original_netflix_dataset")
    
    fig = plt.figure(figsize=(10, 5))

    fig.add_subplot(1, 2, 1)
    sns.boxplot(netflix_df["release_year"])
    plt.ylabel("Release Year")
    plt.title("Boxplot for Release Year with Outliers")

    netflix_df = netflix_df[netflix_df["release_year"] > 2008].copy(deep=True)
    netflix_df.reset_index(inplace=True)

    fig.add_subplot(1, 2, 2)
    sns.boxplot(netflix_df["release_year"])
    plt.ylabel("Release Year")
    plt.title("Boxplot for Release Year without Outliers")

    fig.savefig("static/release_year.png")

    netflix_df["duration_season"] = 0
    netflix_df["duration_movie"] = 0

    for i in range(len(netflix_df)):
        value = netflix_df["duration"][i]
        if ("Seasons" in value):
            netflix_df["duration"][i] = netflix_df["duration"][i].replace("Seasons", "")
            netflix_df["duration_season"][i] = 1
        elif ("Season" in value):
            netflix_df["duration"][i] = netflix_df["duration"][i].replace("Season", "")
            netflix_df["duration_season"][i] = 1
        elif "min" in value:
            netflix_df["duration"][i] = netflix_df["duration"][i].replace("min", "")
            netflix_df["duration_movie"][i] = 1
        else:
            print("Its something new kind of content")
    
    netflix_df["duration"] = netflix_df["duration"].astype(int)
    netflix_df.to_csv("static/cleaned_netflix_dataset.csv")
    netflix_df.to_html("templates/cleaned_netflix_dataset.html", table_id="cleaned_netflix_dataset")
    return redirect("phase1_task2")

def binance_ticker_24hr(request):
    return render(request, "binance_ticker_24hr.html")

def original_netflix_dataset(request):
    return render(request, "original_netflix_dataset.html")

def cleaned_netflix_dataset(request):
    return render(request, "cleaned_netflix_dataset.html")