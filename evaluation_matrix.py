import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split

def evaluate_models():
    st.title("Model Evaluation Results")

    # Load dataset
    dataset = pd.read_csv("Thyroid_train.csv")

    # Define features and target variable
    x = dataset[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
    y = dataset['diagnosis']

    # Split dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # Iterate over models
    for model_name, model in models.items():
        # Fit the model using the training data
        model.fit(x_train, y_train)

        # Make predictions for both training and testing sets
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Calculate evaluation metrics for the training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred)

        # Calculate evaluation metrics for the testing set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)

        # Display evaluation metrics
        st.subheader(f"Model: {model_name}")

        st.write('Model performance for Training set')
        st.write("- Accuracy: {:.4f}".format(train_accuracy))
        st.write("- F1 Score: {:.4f}".format(train_f1))
        st.write("- Precision: {:.4f}".format(train_precision))
        st.write("- Recall: {:.4f}".format(train_recall))
        st.write("- ROC AUC Score: {:.4f}".format(train_roc_auc))
        st.write('-----------------------------------------------------------')

        st.write('Model performance for Test set')
        st.write("- Accuracy: {:.4f}".format(test_accuracy))
        st.write("- F1 Score: {:.4f}".format(test_f1))
        st.write("- Precision: {:.4f}".format(test_precision))
        st.write("- Recall: {:.4f}".format(test_recall))
        st.write("- ROC AUC Score: {:.4f}".format(test_roc_auc))
        st.write('-----------------------------------------------------------')

# Call the function
evaluate_models()
