import streamlit as st
from multiapp import MultiApp

# Add the path to your module directory

# Now you can import your module
import myapp2,home,confusion_matrix,data,EDA,ttest,evaluation_matrix

main = MultiApp()




# Add all your application here
main.add_app("Home",home.show_description)
main.add_app("Model",myapp2.thyroid_cancer_prediction_app)
main.add_app("Model Performace Result",evaluation_matrix.evaluate_models)
main.add_app("Confusion Matrix",confusion_matrix.visualize_confusion_matrix)
main.add_app("T-Tsest",ttest.visualize_t_test_results)
main.add_app("Data Analysis",EDA.visualize_thyroid_data)
main.add_app("Dataset", data.show_data)
main.run()
