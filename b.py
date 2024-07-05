from click import option
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functools import partial
import re
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,roc_auc_score
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, StratifiedKFold
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df_user = pd.read_csv('df_user.csv')
df_keys = pd.read_csv('df_keys.csv')

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
    st.title('#  Exploratory Data Analysis (EDA)')

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



        