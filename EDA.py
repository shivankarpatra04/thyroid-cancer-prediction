import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def visualize_thyroid_data():
    st.title("Thyroid Cancer Dataset Analysis")

    # Load dataset
    ThyroidCancerDF = pd.read_csv("Thyroid_train.csv")

    # Replace diagnosis labels
    ThyroidCancerDF['diagnosis'] = ThyroidCancerDF['diagnosis'].replace({0: 'malignant', 1: 'benign'})

    # Extract features (independent variables) and target variable
    x = ThyroidCancerDF[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
    y = ThyroidCancerDF['diagnosis']

    clf = RandomForestClassifier()
    clf.fit(x, y)
    feature_importances = pd.DataFrame(clf.feature_importances_, index=x.columns, columns=['Importance'])
    

    # Display dataset info, keys, and description

    st.subheader("Dataset Keys:")
    st.write(ThyroidCancerDF.keys())

    st.subheader("Dataset Description:")
    st.write(ThyroidCancerDF.describe())


    # Display statistics for benign and malignant cases
    st.subheader("Benign Statistics:")
    st.write(ThyroidCancerDF[ThyroidCancerDF['diagnosis'] == 'benign'].describe())

    st.subheader("Malignant Statistics:")
    st.write(ThyroidCancerDF[ThyroidCancerDF['diagnosis'] == 'malignant'].describe())

    st.subheader("Feature Importances:")
    st.write(feature_importances)

    st.title("Data Visualization")
    st.markdown("&nbsp;")
    # Custom palette
    custom_palette = ['#ff4f00', '#0a83c2']
    
    st.subheader("Pairplot")
    sns.pairplot(ThyroidCancerDF, hue='diagnosis', palette=custom_palette,vars=['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness'])
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    showPyplotGlobalUse = falseshowPyplotGlobalUse = False

    st.markdown("&nbsp;")
    st.markdown("---")

    st.subheader("Countplot")
    fig, ax = plt.subplots()
    sns.countplot(x='diagnosis', data=ThyroidCancerDF, hue='diagnosis', palette=custom_palette, ax=ax)
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Diagnosis Categories')
    sns.set_style(style='whitegrid')
    st.pyplot(fig)

    st.markdown("&nbsp;")
    st.markdown("---")

    st.subheader("Stripplot")
    features = ['mean_area', 'mean_radius', 'mean_perimeter', 'mean_texture', 'mean_smoothness']

    for feature in features:
        # Create a new figure for each stripplot
        fig, ax = plt.subplots()  # Clears the plot for each iteration
        sns.stripplot(y=feature, data=ThyroidCancerDF, hue="diagnosis", palette=custom_palette, ax=ax)
        ax.set_xlabel('Mean Value')
        ax.set_ylabel('Diagnosis')  # y-axis now shows diagnosis
        ax.set_title(f'Distribution of {feature} by Diagnosis')  # Dynamic title
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("&nbsp;")
    

    st.markdown("&nbsp;")
    st.markdown("---")

    st.subheader("Barplot")
    for feature in features:
        # Create a new figure for each barplot
        fig, ax = plt.subplots()  # Clears the plot for each iteration
        sns.barplot(y=feature, data=ThyroidCancerDF, hue="diagnosis", palette=custom_palette, ax=ax)
        ax.set_xlabel('Mean Value')
        ax.set_ylabel('Diagnosis')  # y-axis now shows diagnosis
        ax.set_title(f'Distribution of {feature} by Diagnosis')  # Dynamic title
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("&nbsp;")
    
    st.markdown("&nbsp;")
    st.markdown("---")

    st.subheader("Outliers for Malignant Diagnosis")
    for feature in features:
        # Create a new figure for each barplot
        miglant_df = ThyroidCancerDF[ThyroidCancerDF['diagnosis'] == 'malignant']
        fig, ax = plt.subplots()  # Clears the plot for each iteration
        sns.boxplot(y=feature, data=miglant_df, hue="diagnosis",palette='dark', ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Malignant')  # y-axis now shows diagnosis
        ax.set_title(f'Distribution of {feature} by Malignant Diagnosis')  # Dynamic title
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("&nbsp;")
    
    st.markdown("&nbsp;")
    st.markdown("---")

    st.subheader("Outliers for Benign Diagnosis")
    for feature in features:
        # Create a new figure for each barplot
        benign_df = ThyroidCancerDF[ThyroidCancerDF['diagnosis'] == 'benign']
        fig, ax = plt.subplots()  # Clears the plot for each iteration
        sns.boxplot(y=feature, data=benign_df, hue="diagnosis",palette='dark', ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Benign')  # y-axis now shows diagnosis
        ax.set_title(f'Distribution of {feature} by Benign Diagnosis')  # Dynamic title
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("&nbsp;")
    
    st.markdown("&nbsp;")
    st.markdown("---")

# Example usage
if __name__ == "__main__":
    visualize_thyroid_data()
