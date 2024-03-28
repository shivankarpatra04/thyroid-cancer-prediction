import streamlit as st
import pandas as pd

def show_data():
    st.title("Thyroid Cancer Dataset")
    st.subheader("Dataset")
    
    # Load the dataset
    df = pd.read_csv("Thyroid_train.csv")
    
    # Display the dataset
    st.write(df)
    st.subheader("About")
    st.write("""
    **Dataset Features:**
    - **Mean Radius**: Represents the average size or radius of thyroid nodules or tumors in individuals. It can provide insights into the physical characteristics of thyroid growths.
    - **Mean Texture**: Texture in this context may describe the structural properties or consistency of the thyroid tissue. It could help in identifying irregularities.
    - **Mean Perimeter**: Perimeter refers to the measurement of the outer boundary of thyroid growths. This feature offers information about the shape and extent of thyroid nodules.
    - **Mean Area**: The area feature likely represents the surface area of thyroid growths. It is a crucial parameter for evaluating the size and extent of thyroid tumors.
    - **Mean Smoothness**: Smoothness could indicate the evenness or regularity of thyroid tissue. It may help in identifying irregularities or variations in the tissue structure.
    - **Diagnosis**: The diagnosis feature serves as the target variable. It categorizes individuals into classes, typically "benign" and "malignant." This is the label that machine learning models aim to predict.
    """)

# Example usage
if __name__ == "__main__":
    show_data()

