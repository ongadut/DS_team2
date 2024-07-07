
This Python script implements a machine learning model to detect Parkinson's disease based on keystroke data. It leverages various libraries such as Streamlit for a user-friendly interface, Pandas for data manipulation and Shap for model interpretability.
Data and Preprocessing
User data: Contains general and medical information for each user, including demographic, medical history, and Parkinson's disease status.
Keystroke data: Contains detailed keystroke metrics like hold time, latency time, and flight time for each user.
Data is preprocessed, merged, and cleaned to prepare for model training.
Model Development
Exploratory Data Analysis (EDA): Visualizations are used to understand data distributions, relationships between features, and potential correlations with Parkinson's disease.
Model Selection: The script provides options for Random Forest, Bernoulli Naive Bayes, Logistic Regression, and Catboost models.
Model Training and Evaluation: Models are trained on a portion of the data and evaluated using metrics like accuracy, F1-score, and AUC.
Model Interpretability: Shap values are used to explain the impact of features on model predictions.
Usage
Download the data:
You can download the files from here:
https://www.mediafire.com/file/ytinaz8nsapy6j9/ArchivedUsers.zip/file
https://www.mediafire.com/file/d7s40qiglfbs8xt/TappyData.zip/file
Install required libraries
Prepare data:
First start whit Final_Project_Team2_DataLoad.ipynb to load the data. note change the path
Second continue whit Final_Project_Team2_FeatureEngineering.ipynb
Final_Project_Team2_FeatureEngineering_FlightTime.ipynb
Final_Project_Team2_FeatureEngineering_LDA.ipynb
Third run b.py scrept for Streamlit app

