import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def visualize_confusion_matrix():
    # Load data
    ThyroidCancerDF = pd.read_csv("Thyroid_train.csv")
    x = ThyroidCancerDF[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
    y = ThyroidCancerDF['diagnosis']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    # Create RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    # Predict
    y_test_predicted = clf.predict(x_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_predicted)

    # Normalize the confusion matrix (optional)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Streamlit app
    st.title('Confusion Matrix Visualization')

    # Display the confusion matrix as a heatmap
    st.write('### Confusion Matrix')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Add text for each cell (optional)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.xticks(range(len(clf.classes_)), clf.classes_, rotation=45)
    plt.yticks(range(len(clf.classes_)), clf.classes_)
    plt.tight_layout()

    st.pyplot(plt)

    # Show the raw data
    st.write('### Raw Confusion Matrix Data')
    st.write(pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_))

# Example usage
if __name__ == '__main__':
    visualize_confusion_matrix("Thyroid_train.csv")
