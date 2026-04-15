import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from helper import * 

st.title('DS4420 Pet Adoption!')

@st.cache_data
def load_pet_pref_data():
    pet_preferences = pd.read_csv("pet_questionaire.csv")
    return pet_preferences

pet_preferences = load_pet_pref_data()

@st.cache_data
def load_clean_pet_pref_data(): 
    pet_preferences_clean = pet_preferences.dropna()
    return pet_preferences_clean


landing, interactive_charts, models, about = st.tabs(["Home Page", "Interactive charts", "Our Models", "About"])


# LANDING PAGE 
with landing:
    st.write("Welcome to our Pet adoption project! This project aims to help understand people's preferences for pets, and then also try to understand if pet features have impact on adoption speed.")
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.image("petfinder-adoption-prediction_data/test_images/0a3d2b273-1.jpg")

# CHARTS
with interactive_charts:
    st.header("Interactive Time Series Plots")
    st.subheader("Try adjusting these settings and see what happens! ")

    
    # numlags = st.slider(
    #     "Number of Lags (recommended to sit around 51)",
    #     min_value=3,
    #     max_value=90,
    #     value=51,
    #     step=1
    # )

    # weather_var = st.selectbox(
    #     "Select Weather Variable",
    #     ["TAVG", "PRCP", "TMIN", "TMAX"], 
    #     index=0
    # )

    # st.write("Building a interactive model with ", numlags, "lags and ",weather_var," weather type")
    # with st.spinner(f"Running model using '{weather_var}'..."):
    #     preds, test, beta = run_forecast_pipeline(
    #         combined,
    #         numlags=numlags,
    #         weather_var=weather_var
    #     )

    # fig, ax = plt.subplots(figsize=(14,6))

    # ax.plot(test['DATE'], test['consumption'], label="Actual", linewidth=2)
    # ax.plot(test['DATE'], preds.values, label="Predicted", linestyle="--")

    # ax.set_title(f"Forecast with {numlags} Lags — Weather: {weather_var}")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Consumption")
    # ax.legend()
    # ax.grid(True)

    # st.pyplot(fig)

    # actuals = test['consumption'].values
    # rmse = np.sqrt(np.mean((actuals - preds.values)**2))
    # mae = np.mean(np.abs(actuals - preds.values))
    # mape = np.mean(np.abs((actuals - preds.values) / actuals)) * 100

    # st.subheader("Model Performance")
    # st.write(f"**RMSE:** {rmse:.2f}")
    # st.write(f"**MAE:**  {mae:.2f}")
    # st.write(f"**MAPE:** {mape:.2f} %")



with models:

    with st.spinner(f"Running MLP model:"):
        y_test_pred, y_test = run_mlp_pipeline(pet_preferences)

    test_r2 = r2_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Pet', 'Has Pet'],
                yticklabels=['No Pet', 'Has Pet'],
                ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')


    st.pyplot(fig)

    # pp_check, comp = st.tabs(['PP Check', 'Comparison'])
    # with pp_check:
    #     # st.image("plots/pp_check.png")
    #     st.text("Distribution of actual test values (black) vs posterior predictions of test values (light blue)")
    # with comp:
    #     # st.image("plots/pred_vs_actual.png")
    #     st.text("Posterior prediction test means with 95% Confidence Interval vs actual test values")

# About us 
with about:
    st.write("This is about us")