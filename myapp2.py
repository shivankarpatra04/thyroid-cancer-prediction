import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.write("""
# Thyroid Cancer Prediction App

Assess thyroid nodules to determine cancer risk through thorough analysis, aiding in early detection and appropriate treatment planning.

Data obtained from the [Kaggle](https://www.kaggle.com/datasets/bibin2001/thyroid-cancer-predition) by PRAKASH NAMBIAR.
""")
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://drive.google.com/file/d/1RRRD9cJqKJp5QSeRPt0Lz8sON7EtQXnu/view?usp=sharing)
""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        """Gets user input features."""
        mean_radius = st.sidebar.slider('Mean Radius', min_value=0.0, max_value=35.0, value=15.0)
        mean_texture = st.sidebar.slider('Mean Texture', min_value=0.0, max_value=50.0, value=25.0)
        mean_perimeter = st.sidebar.slider('Mean Perimeter', min_value=0.0, max_value=200.0, value=75.0)
        mean_area = st.sidebar.slider('Mean Area', min_value=0.0, max_value=3000.0, value=1500.0)
        mean_smoothness = st.sidebar.slider('Mean Smoothness', min_value=0.0, max_value=0.5, value=0.2)

        data = {'mean_radius': mean_radius,
                'mean_texture': mean_texture,
                'mean_perimeter': mean_perimeter,
                'mean_area': mean_area,
                'mean_smoothness': mean_smoothness
                }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

st.subheader('User Input features')

ThyroidCancerDF = pd.read_csv("Thyroid_train.csv")

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# Preprocess diagnosis column (if diagnosis is 'M' or 'B')
ThyroidCancerDF['diagnosis'] = ThyroidCancerDF['diagnosis'].replace({'malignant': 0, 'benign': 1})

X = ThyroidCancerDF[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
Y = ThyroidCancerDF['diagnosis']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

st.subheader('Prediction')
if prediction[0] == 0:
    st.write('Malignant')
else:
    st.write('Benign')

st.subheader('Prediction Probability')
st.write(prediction_proba)
