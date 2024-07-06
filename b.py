from click import option
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import Pool
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from streamlit_option_menu import option_menu
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from streamlit_shap import st_shap
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings("ignore")

df_user = pd.read_csv('df_user.csv')
df_keys = pd.read_csv('df_keys.csv')
full_set = pd.read_csv('full_set.csv')
full_set_LDA = pd.read_csv('full_set_LDA.csv')
full_set_FligthTime = pd.read_csv('full_set_FligthTime.csv')

full_set1 = pd.merge(full_set, df_user, on='ID', how='inner')
new_df = full_set1.drop([ 'BirthYear', 'Gender', 
       'DiagnosisYear', 'Parkinsons_x'], axis=1)

with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu',
        options=['Home','Exploration Data','Prediction','About as']
        )

if selected == 'Home':
    st.title("***Project - Detection of Parkinson’s disease with keystroke data*** ")
    st.write('The idea is to create an ML model that will detect whether a user has Parkinson’s Disease or not based on their keystroke data.') 
    st.write('**User data** (ArchivedUsers.zip): ***Contains text files with general and medical information for each user.***')
    st.write('The fields are:')
    st.write('● Birth Year: Year of birth')
    st.write('● Gender: Male/Female')
    st.write('● Parkinsons: Whether they have Parkinsons Disease [True/False]')
    st.write('● Tremors: Whether they have tremors [True/False]')
    st.write('● Diagnosis Year: If they have Parkinsons, when was it first diagnosed')
    st.write('● Whether there is sidedness of movement [Left/Right/None] (self reported)')
    st.write('● UPDRS: The UPDRS score (if known) [1 to 5]')
    st.write('● Impact: The Parkinsons disease severity or impact on their daily life [Mild/None] (self reported)')
    st.write('● Levadopa: Whether they are using Sinemet and the like [Yes/No]')
    st.write('● DA: Whether they are using a dopamine agonist [Yes/No]')
    st.write('● MAOB: Whether they are using an MAO-B inhibitor [Yes/No]')
    st.write('● Other: Whether they are taking another Parkinsons medication [Yes/No]<br>')
    st.write('**Keystroke data** (TappyData.zip): ***Contains text files with records of keystroke data over period of time for each user***.')
    st.write('The fields are:')
    st.write('● UserKey: 10 character code for that user')
    st.write('● Date: YYMMDD')
    st.write('● Timestamp: HH:MM:SS:SSS')
    st.write('● Hand: L or R key pressed')
    st.write('● Hold time: Time between press and release for current key mmmm.m milliseconds')
    st.write('● Direction: Previous to current LL, LR, RL, RR (and S for a space key)')
    st.write('● Latency time: Time between pressing the previous key and pressing the current key. Milliseconds')
    st.write('● Flight time: Time between release of previous key and press of current key. Milliseconds')
    st.write('keystroke data captures detailed metrics such as timing, hand usage, and transition patterns, which are crucial for identifying subtle motor control issues associated with Parkinsons Disease. The time metrics, including hold time, latency time, and flight time, are especially important as they provide precise measurements of typing speed and rhythm, revealing delays and irregularities indicative of motor impairments. By analyzing these variations in typing behavior, the data allows for the detection of asymmetries and disruptions that reflect the characteristic symptoms of the disease.')
if selected == 'Exploration Data':
    st.title('Exploratory Data Analysis (EDA)')
    # Pie Chart
    male_parkinsons_count = df_user[(df_user['Parkinsons'] == True) & (df_user['Gender'] == 'Male')]['Parkinsons'].count()
    male_non_parkinsons_count = df_user[(df_user['Parkinsons'] == False) & (df_user['Gender'] == 'Male')]['Parkinsons'].count()
    female_parkinsons_count = df_user[(df_user['Parkinsons'] == True) & (df_user['Gender'] == 'Female')]['Parkinsons'].count()
    female_non_parkinsons_count = df_user[(df_user['Parkinsons'] == False) & (df_user['Gender'] == 'Female')]['Parkinsons'].count()
    total_male_count = male_parkinsons_count + male_non_parkinsons_count
    total_female_count = female_parkinsons_count + female_non_parkinsons_count
    total_labels = ['Male', 'Female']
    total_values = [total_male_count, total_female_count]
    fig_total = px.pie(names=total_labels, values=total_values, title='Total Distribution by Gender (Parkinsons and Non-Parkinsons)',
                    color=total_labels, color_discrete_sequence=['#d62728', '#1f77b4'], opacity=0.8)
    labels = ['Male with Parkinsons', 'Female with Parkinsons', 'Male without Parkinsons', 'Female without Parkinsons']
    values = [male_parkinsons_count, female_parkinsons_count, male_non_parkinsons_count, female_non_parkinsons_count]
    color_map = {
        'Male with Parkinsons': '#1f77b4',
        'Male without Parkinsons': '#95b3d7',
        'Female with Parkinsons': '#d62728',
        'Female without Parkinsons': '#ff9896'
    }
    fig_parkinsons = px.pie(names=labels, values=values, title='Distribution of Parkinsons by Gender',
                            color=labels, color_discrete_map=color_map, opacity=0.8, hole=0.4)
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                        subplot_titles=['Gender Distribution %', 'Parkinsons Distribution by Gender %'])
    fig.add_trace(fig_total.data[0], row=1, col=1)
    fig.add_trace(fig_parkinsons.data[0], row=1, col=2)
    fig.update_traces(rotation=-180, selector=dict(row=1, col=1, type='pie'))

    fig.update_layout(title_text='Comparison of Gender Distributions')
    st.plotly_chart(fig)
    # Bar chart
    parkinsons_true_df = df_user[df_user['Parkinsons'] == True]
    combined_drug_data = parkinsons_true_df.melt(id_vars=['Tremors'], 
                                                    value_vars=['Levadopa', 'DA', 'MAOB', 'Other'], 
                                                    var_name='Drug', value_name='Usage')
    tremor_drug_count = combined_drug_data.groupby(['Tremors', 'Drug', 'Usage']).size().reset_index(name='Count')
    combined_drug_data = tremor_drug_count
    fig_combined = px.bar(combined_drug_data, x='Tremors', y='Count', color='Usage', barmode='group',
                        facet_col='Drug', title='Eefficacy of separate drugs on Tremors',
                        category_orders={'Tremors': [False, True], 'Usage': [False, True]})


    fig_combined.update_layout(xaxis_title='Tremor Status', yaxis_title='Count', legend_title='Drug Usage')
    st.plotly_chart(fig_combined)
    # Histogram of Age
    df_user['Age'] = df_user['DiagnosisYear'] - df_user['BirthYear']
    mean_age = df_user['Age'].mean()
    parkinsons_data = df_user[df_user['Parkinsons'] == True]
    fig_age = px.histogram(parkinsons_data, x='Age', title='Distribution of Age for Parkinsons Patients',
                    labels={'Age': 'Age', 'count': 'Number of Patients'},
                    nbins=10,  
                    marginal='rug',  
                    opacity=0.9,  
                    color_discrete_sequence=['#1f77b4'])  
    fig_age.add_vline(x=mean_age, line_dash="dash", line_color="red", 
                annotation_text=f'Mean Age: {mean_age:.1f}', annotation_position="right", 
                annotation_font=dict(size=12, color='red'))
    fig_age.update_layout(xaxis_title='Age', yaxis_title='Number of Patients',
                    bargap=0.1,  
                    showlegend=False)
    st.plotly_chart(fig_age)

    df_user['CombinedDrugs'] = df_user[['Levadopa', 'DA', 'Other', 'MAOB']].any(axis=1)


    parkinsons_df = df_user[df_user['Parkinsons'] == True]


    grouped_df = parkinsons_df.groupby(['CombinedDrugs', 'Tremors']).size().reset_index(name='Count')
    grouped_df['CombinedDrugs'] = grouped_df['CombinedDrugs'].map({True: 'Using Drugs', False: 'Not Using Drugs'})
    grouped_df['Tremors'] = grouped_df['Tremors'].map({True: 'Tremors', False: 'No Tremors'})


    fig = px.bar(grouped_df, 
                x='CombinedDrugs', 
                y='Count', 
                color='Tremors', 
                barmode='stack',
                title='Effect of Combined Drug Use on Tremors for Patients with Parkinson\'s',
                labels={'Count': 'Number of Patients', 'CombinedDrugs': 'Drug Use'})

    st.plotly_chart(fig)

    ##key data exploration
    st.title('Key Data Exploration')
    data=new_df.drop(['ID', "mean_RR", "mean_LL", "mean_RL", "mean_LR",
                  'Tremors', 'Sided', 'Impact', 'Levadopa', 'DA', 'MAOB',
       'Other'], axis=1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), sharex=False, sharey=False)
    axes = axes.ravel()
    cols= ["mean_R", "mean_L"]

    datan = pd.DataFrame
    for col, ax in zip(cols, axes):
        datan = data[col] 
        sns.histplot(data=data, x=col, ax=ax, bins=20, kde=True,  legend=True, hue="Parkinsons_y", 
                    fill=True, common_norm=False, palette ="BuPu", alpha=.6, linewidth=0)
        ax.set(title=f'Distribution of Column: {col}', xlabel=None)
        
    fig.tight_layout()
    st.pyplot(fig)

    data=new_df.drop(['ID', 'mean_L', 'mean_R', 
                  'Tremors', 'Sided', 'Impact', 'Levadopa', 'DA', 'MAOB',
       'Other'], axis=1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 7), sharex=False, sharey=False)
    axes = axes.ravel()
    cols= ["mean_RR", "mean_LL", "mean_RL", "mean_LR"]
    print(cols)
    datan = pd.DataFrame
    for col, ax in zip(cols, axes):
        datan = data[col] 
        sns.kdeplot(data=data, x=col, ax=ax, hue="Parkinsons_y",  fill=True, common_norm=False, palette ="BuPu", alpha=.5, linewidth=0)
        ax.set(title=f'Distribution of Column: {col}', xlabel=None)
        
    fig.tight_layout()
    st.pyplot(fig)


if selected == 'Prediction':
    st.title('Prediction scores for model')
    option = st.selectbox(
    "Select model",
    ("RandomForest", "BernoulliNB",'LogisticRegression', "Catboost"))
if option == 'RandomForest':
    st.write('RandomForest feature:\n',full_set.columns[1:6])
    st.write('#RandomForest parametars:\n','\nn_estimators: 500\n','\nmax_depth: 5\n')
    

    rf = pickle.load(open('RandomForestClassifier.sav', 'rb'))
    X = full_set.drop(columns=['Parkinsons', 'ID'], axis=1)
    y = full_set['Parkinsons']
    cv = cross_val_score(rf, X, y, cv=5)
    
    st.write(f'Cros validation score: {cv.mean():.2f}')

    st.write('Shap explanation')
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer(X)
    st_shap(shap.plots.beeswarm(shap_values[:,:,1], max_display=10))
    value = st.slider("Choose a user", 1, 80, 6)
    st_shap(shap.plots.waterfall(shap_values[value,:,0], max_display=10))
if option == 'Catboost':
    X_train, X_test, y_train, y_test = train_test_split(full_set_LDA.drop(columns=['Parkinsons']), full_set['Parkinsons'], test_size=0.2, random_state=42)
    lda_model = pickle.load(open('LDA_model.sav', 'rb'))
    st.write('Data is splited Train : 80% Test: 20%')
    st.write('For tunig the model use LDA(Linear discriminant analysis)') 
    st.write('Catboost feature:',full_set_LDA.columns[:-1])
    st.write('Catboost parametars: learning_rate= 0.03\n,\niterations=1000\n, \ndepth=8\n, \nloss_function=Logloss\n, \nmin_data_in_leaf=5\n')
    model_cat = pickle.load(open('CatBoostClassifier.sav', 'rb')) 
    X_train = lda_model.transform(X_train)
    X_test = lda_model.transform(X_test)
    train_data = Pool(data=X_train, label=y_train)
    test_data = Pool(data=X_test, label=y_test)
    
    y_pred = model_cat.predict(test_data)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred)      
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"F1-Score: {f1:.3f}")
    st.write(f'AUC score: {auc:.3f}')
    evals_result = model_cat.get_evals_result()
    train_loss = evals_result['learn']['Logloss']
    test_loss = evals_result['validation']['Logloss']
    iterations = np.arange(1, len(train_loss) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(iterations, train_loss, label='Training Loss', color='blue')
    plt.plot(iterations, test_loss, label='Validation Loss', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('CatBoost Training Progress')
    plt.legend()
    plt.grid()
    st.pyplot(plt)
if option == 'BernoulliNB':
    st.write('Bernoulli distribution is used for discrete probability calculation. It either calculates success or failure the features are independent of one another')
    st.write('Data is splited Train : 80% Test: 20%')
    st.write('BernoulliNB feature:',full_set_FligthTime.columns[:-1])
    st.write('BernoulliNB parametars: alpha= 0.04, fit_prior=True')
    X = full_set_FligthTime.drop(columns=['Parkinsons', 'ID'], axis=1)
    y = full_set_FligthTime['Parkinsons']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_ber=X_train.values
    y_ber = y_train.values
    X_test_ber=X_test.values
    y_test_ber = y_test.values

    bnb = pickle.load(open('BernoulliNB.sav', 'rb'))

    y_pred_ber = bnb.predict(X_test_ber)
    pred_ber = bnb.predict_proba(X_test_ber)
    auc = roc_auc_score(y_test_ber, pred_ber[:, 1])
    accuracy = accuracy_score(y_test, y_pred_ber)
    f1 = f1_score(y_test, y_pred_ber, average='weighted')
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"F1-Score: {f1:.3f}")
    st.write(f'AUC score: {auc:.3f}')
    
if option == 'LogisticRegression':
    st.write('Data is splited Train : 80% Test: 20%')
    st.write('LogisticRegression feature:',full_set_FligthTime.columns[1:-1])
    st.write('LogisticRegression parametars:')
    X = full_set_FligthTime.drop(columns=['Parkinsons', 'ID'], axis=1)
    y = full_set_FligthTime['Parkinsons']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_log= scaler.fit_transform(X_train)
    X_test_log = scaler.transform(X_test)

    model_Logistic = pickle.load(open('LogisticRegression.sav', 'rb'))
    y_pred_log = model_Logistic.predict(X_test_log)
    auc = roc_auc_score(y_test,y_pred_log )
    accuracy = accuracy_score(y_test, y_pred_log)
    f1 = f1_score(y_test, y_pred_log, average='weighted')
        
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"F1-Score: {f1:.3f}")
    st.write(f'AUC score: {auc:.3f}')





    


    



        