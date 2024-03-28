import streamlit as st

def show_description():
    st.title("Welcome to Thyroid Cancer Prediction Project")

    st.write("""
    Thyroid cancer is a significant health concern globally, and early detection plays a crucial role in improving treatment outcomes. The "Thyroid Cancer Prediction" project is designed to leverage machine learning techniques to assess thyroid nodules and predict cancer risk accurately. By analyzing a comprehensive dataset comprising clinical and demographic features, our goal is to aid in early detection, facilitate appropriate treatment planning, and ultimately enhance patient outcomes.

    **Key Objectives:**
    - **Early Detection**: Identify individuals at risk of thyroid cancer at an early stage.
    - **Accurate Prediction**: Develop machine learning models to predict cancer risk with high accuracy.
    - **Informative Analysis**: Explore the dataset to gain insights into the factors influencing thyroid cancer.

    **Pages Available:**

    - **Model**: Build and evaluate machine learning models for thyroid cancer prediction.
    - **Model Performance Result**: View the performance metrics of different models.
    - **Confusion Matrix**: Visualize confusion matrices for model evaluation.
    - **T-Test**: Perform statistical analysis using t-tests to compare groups.
    - **Data Analysis**: Explore and analyze the dataset containing clinical and demographic features.
    - **Dataset**: View the dataset used for the project.

    We invite you to explore the various sections of our project to gain insights into thyroid cancer prediction and contribute to the advancement of medical science.
    """)


def main():
    show_description()

if __name__ == "__main__":
    main()