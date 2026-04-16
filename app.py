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

@st.cache_data
def load_pet_data(): 
    pets = pd.read_csv("bayes r/pet_adoption_data.csv")
    return pets

pets = load_pet_data()

landing, interactive_charts, models, about = st.tabs(["Home Page", "Interactive charts", "Our Models", "About"])


# LANDING PAGE 
with landing:
    st.header("Barkesian Models")
    st.write("Pet ownership has long been associated with positive outcomes, from teaching children the responsibilities of caring for another life to combating stress and mental illnesses to working as service animals. But certain pets work better for people’s lifestyles, household dynamics, and personality traits. Our project aims to create an MLP predictive model that reads in a person’s personality traits and matches them to the type of pet that best fits them, a CNN classifier that identifies an adoptable animal’s species and key traits, and a Bayesian predictive model that predicts the likelihood of pet adoption based on the animal’s identified key traits.")
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.image("petfinder-adoption-prediction_data/test_images/0a3d2b273-1.jpg")

# CHARTS
with interactive_charts:

    st.header("Interactive Adoption Plots")
    st.subheader("Explore how different features affect adoption outcomes")
    st.write("Note: 0 on the x axis represents false and 1 is true")

    # controls
    feature = st.selectbox(
        "Select Feature to Analyze",
        ["PetType", "Breed", "Vaccinated", "HealthCondition", "PreviousOwner"],
        index=0
    )

    plot_type = st.radio(
        "Select Plot Type",
        ["Adoption Rate", "Counts"],
        horizontal=True
    )

    pet_filter = st.selectbox(
        "Filter by Pet Type",
        ["All"] + sorted(pets["PetType"].unique()),
        index=0
    )

    filtered = pets.copy()

    if pet_filter != "All":
        filtered = filtered[filtered["PetType"] == pet_filter]

    if feature == "Breed":
        top_vals = filtered["Breed"].value_counts().nlargest(10).index
        filtered = filtered[filtered["Breed"].isin(top_vals)]

    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_type == "Adoption Rate":
        grouped = (
            filtered
            .groupby(feature)["AdoptionLikelihood"]
            .mean()
            .sort_values(ascending=False)
        )

        ax.bar(grouped.index, grouped.values)
        ax.set_ylabel("Adoption Rate")
        ax.set_title(f"Adoption Rate by {feature}")

    else: 
        counts = (
            filtered
            .groupby([feature, "AdoptionLikelihood"])
            .size()
            .unstack(fill_value=0)
        )

        counts.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Count")
        ax.set_title(f"Adoption Counts by {feature}")
        ax.legend(["Not Adopted", "Adopted"])

    ax.set_xlabel(feature)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig)

with models:
    with st.spinner(f"Fetching predictions..."):
        y_test_pred, y_test = run_mlp_pipeline(pet_preferences)


    test_r2 = r2_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)

    st.header(f"MLP Model Performance")
    st.write(f"R2 Score: {test_r2:.2f}")
    st.write(f"Accuracy: {accuracy*100:.2f}%")


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