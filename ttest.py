import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind

def visualize_t_test_results():
    st.title("T-Test Results for Thyroid Cancer Dataset")

    # Load dataset
    ThyroidCancerDF = pd.read_csv("Thyroid_train.csv")

    # Define groups based on diagnosis
    ThyroidCancerDF['diagnosis'] = ThyroidCancerDF['diagnosis'].replace({0: 'malignant', 1: 'benign'})
    benign_group = ThyroidCancerDF[ThyroidCancerDF['diagnosis'] == 'benign']
    malignant_group = ThyroidCancerDF[ThyroidCancerDF['diagnosis'] == 'malignant']

    # Features of interest
    features = ['mean_radius', 'mean_area', 'mean_texture', 'mean_perimeter', 'mean_smoothness']

    # Perform t-tests directly between groups
    for feature in features:
        t_statistic, p_value = ttest_ind(a=benign_group[feature], b=malignant_group[feature], equal_var=False)

        st.subheader(f"T-Test Results for {feature}")
        st.write(f"t-statistic: {t_statistic:.4f}")
        st.write(f"p-value: {p_value:.4f}")

        # Interpretation
        if p_value < 0.05:
            st.write(f"Hypothesis Rejected (H1): {feature} is significantly different between benign and malignant tumors.")
        else:
            st.write(f"Hypothesis Accepted (H0): Insufficient evidence to conclude a significant difference in {feature} for malignant tumors.")

        st.write('-----------------------------------------------------------')

    # Interpretations for the features
    st.subheader("Interpretations:")
    st.write("Features like mean_radius, mean_area, and mean_perimeter likely measure the overall size or shape of the nodule.")
    st.write("These features might show a more straightforward difference, with malignant tumors generally being larger or having a larger perimeter.")

    st.write("It's possible that 'mean_smoothness' and 'mean_texture' capture similar information. Both might reflect the evenness or regularity of the tissue.")
    st.write("In this case, the interpretation for lower mean_smoothness in malignant tumors would be the same as mean_texture - a less smooth or more irregular appearance due to heterogeneity in malignant tissue.")

if __name__ == "__main__":
    visualize_t_test_results()
