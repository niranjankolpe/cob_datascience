from django.shortcuts import render, redirect

import pandas as pd
import numpy as np
import requests

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

def original_netflix_dataset(request):
    return render(request, "original_netflix_dataset.html")

def phase1_task2(request):
    return render(request, "phase1_task2.html")

def phase1_task2_refresh(request):
    netflix_df = pd.read_csv("static/original_netflix_dataset.csv")
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
    netflix_df.to_csv("static/phase1_task1_cleaned_dataset.csv")
    netflix_df.to_html("templates/phase1_task1_cleaned_dataset.html", table_id="phase1_task1_cleaned_dataset")
    return redirect("phase1_task2")

def binance_ticker_24hr(request):
    return render(request, "binance_ticker_24hr.html")

def phase1_task1_cleaned_dataset(request):
    return render(request, "phase1_task1_cleaned_dataset.html")

def phase2_task1(request):
    return render(request, "phase2_task1.html")

def phase2_task1_refresh(request):
    df = pd.read_csv("static/original_netflix_dataset.csv")
    df.to_html("templates/original_netflix_dataset.html")

    for i in range(len(df)):
        df["show_id"][i] = df["show_id"][i].split("s")[1]
    
    for i in range(len(df)):
        if df["type"].iloc[i] == "Movie":
            df["duration"].iloc[i] = df["duration"].iloc[i].replace("min", "")
        elif df["type"].iloc[i] == "TV Show":
            if "Seasons" in df["duration"].iloc[i]:
                df["duration"].iloc[i] = df["duration"].iloc[i].replace("Seasons", "")
            elif "Season" in df["duration"].iloc[i]:
                df["duration"].iloc[i] = df["duration"].iloc[i].replace("Season", "")
    
    df["show_id"] = df["show_id"].astype(int)
    df["duration"] = df["duration"].astype(int)

    df_new = df[df["release_year"] > 2008].copy(deep=True)

    # Displaying the boxplot
    sns.boxplot(df_new["release_year"])
    plt.ylabel("Release Year")
    plt.title("Boxplot for Release Year")
    plt.savefig("static/trial_release_year.png")

    generate_fig(1, df_new)
    generate_fig(2, df_new)
    generate_fig(3, df_new)
    generate_fig(4, df_new)
    generate_fig(5, df_new)
    generate_fig(6, df_new)

    df_new.to_csv("static/phase2_task1_cleaned_dataset.csv")
    df_new.to_html("templates/phase2_task1_cleaned_dataset.html")

    return redirect("phase2_task1")

def generate_fig(i, df1):
    if i==1:
        # Insight No 1: USA has the highest number of movies and series content combined. Followed by India and United Kingdom
        fig1 = sns.barplot(x=list(df1["country"].value_counts().keys())[:10], y=list(df1["country"].value_counts().values[:10]))
        fig1.grid()
        plt.xlabel("Country")
        plt.ylabel("Count of Movies and Series Combined")
        plt.title("Country Vs Count (Movies and Series)")
        plt.savefig("static/insight_1.png")
        del fig1
    elif i==2:
        # Insight No 2: Rapid growth in content release has been observed from 2015 with peak in 2018 and decline thereof.
        fig2 = sns.barplot(x=list(df1["release_year"].value_counts().keys()), y=list(df1["release_year"].value_counts().values))
        plt.savefig("static/insight_2.png")
        del fig2
    elif i==3:
        # Insight No 3: Total 5091 Movies and 2519 Series have been created between the year 2009 and 2021
        fig3 = df1["type"].value_counts().plot(kind="bar")
        fig3.grid()
        plt.ylabel("Count")
        plt.savefig("static/insight_3.png")
        del fig3
    elif i==4:
        # Insight No 4: Most customers are adults since the content rated "TV-MA" tops the list of most popular content based on rating
        fig4 = df1["rating"].value_counts().plot(kind="bar")
        fig4.grid()
        plt.savefig("static/insight_4.png")
        del fig4
    elif i==5:
        # Insight No 5:
        # 1) Rajiv Chilaka has directed the most Movies and Alastair Fothergill has directed the most TV Shows
        # 2) Most directors prefer to direct Movies rather than TV Shows
        fig5 = sns.countplot(data=df1, x="director", hue="type",order=df1.director.value_counts().iloc[1:11].index, palette=["Black", "Red"])
        fig5.grid()
        plt.xticks(rotation=20)
        plt.savefig("static/insight_5.png")
        del fig5
    elif i==6:
        # Insight No 6: The most popular Genre is "International Movies" followed by "Dramas" and "International TV Shows"
        genres = df1.listed_in.str.split(", ").explode()
        fig6 = genres.value_counts()[:10].plot(kind="bar",
                                        color=["red", "green", "blue", "orange", "yellow", "cyan", "brown", "pink", "purple", "maroon"])
        plt.xticks(rotation=35)
        plt.savefig("static/insight_6.png")
        del fig6
    else:
        pass
    return True

def phase2_task1_cleaned_dataset(request):
    return render(request, "phase2_task1_cleaned_dataset.html")

def phase2_task2(request):
    return render(request, "phase2_task2.html")

def phase2_task2_train_dataset(request):
    return render(request, "phase2_task2_train_dataset.html")

def phase2_task2_test_dataset(request):
    return render(request, "phase2_task2_test_dataset.html")

def phase2_task2_model_refresh(request):
    train_df = pd.read_csv("static/P2 T2 Train Dataset.csv")
    train_df.to_html("templates/phase2_task2_train_dataset.html")

    test_df = pd.read_csv("static/P2 T2 Test Dataset.csv")
    train_df.to_html("templates/phase2_task2_test_dataset.html")

    train_df["y"].fillna(0, inplace=True)

    x_train = np.array(train_df.x).reshape(-1, 1)

    x_test = np.array(test_df.x).reshape(-1, 1)
    y_test = np.array(test_df.y)

    model_evaluation = pd.DataFrame(columns=['Model Name', 'R2 Score', 'MAE', 'MSE', 'RMSE'])

    lin_reg = LinearRegression()
    lin_reg.fit(x_train, train_df.y)
    y_pred = lin_reg.predict(x_test)
    with open("static/phase2_task2_linear_regression_model", "wb") as f1:
        joblib.dump(lin_reg, f1)
    model_name = "Linear Regression"
    r2_score_value = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_evaluation.loc[len(model_evaluation)] = [model_name, r2_score_value, mae, mse, rmse]
    del lin_reg

    svr = SVR()
    svr.fit(x_train, train_df.y)
    y_pred = svr.predict(x_test)
    with open("static/phase2_task2_support_vector_regression_model", "wb") as f2:
        joblib.dump(svr, f2)
    model_name = "Support Vector Regression"
    r2_score_value = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_evaluation.loc[len(model_evaluation)] = [model_name, r2_score_value, mae, mse, rmse]
    del svr

    rf_reg = RandomForestRegressor()
    rf_reg.fit(x_train, train_df.y)
    y_pred = rf_reg.predict(x_test)
    with open("static/phase2_task2_random_forest_regression_model", "wb") as f3:
        joblib.dump(rf_reg, f3)
    model_name = "Random Forest Regression"
    r2_score_value = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_evaluation.loc[len(model_evaluation)] = [model_name, r2_score_value, mae, mse, rmse]
    del rf_reg

    model_evaluation.to_html("templates/phase2_task2_model_evaluation.html")

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.title("Model Evaluation")
    sns.barplot(data=model_evaluation, x="Model Name", y="R2 Score", hue='RMSE', palette=["red", "green", "blue"])
    plt.savefig("static/phase2_task2_model_evaluation.png")

    return redirect("home")

def phase2_task2_prediction(request):
    data = dict()
    if request.method == "POST":
        user_input = None
        model_name = None
        prediction = None

        user_input = int(request.POST["user_input"])
        model_name = str(request.POST["model_name"])
        model = joblib.load(str("static/" + model_name))
        if model_name == "phase2_task2_linear_regression_model":
            model_name = "Linear Regression"
        elif model_name == "phase2_task2_support_vector_regression_model":
            model_name = "Support Vector Regression"
        elif model_name == "phase2_task2_random_forest_regression_model":
            model_name = "Random Forest Regression"
        else:
            model_name = "No model"
        prediction = model.predict([[user_input]])
        data = {'user_input': user_input, 'model_name': model_name, 'prediction': str(prediction[0])}
    return render(request, "phase2_task2_prediction.html", data)

def phase2_task2_model_evaluation(request):
    return render(request, "phase2_task2_model_evaluation.html")
