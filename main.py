import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model=joblib.load("rf_personality_synthetic.pkl")
df=pd.read_csv("personality_synthetic_dataset.csv")


def about():
    st.title("🧟 Who's most likely to cancel the plans? ")
    st.header("🤖 About the model:")
    st.markdown('''
            -Welcome to this **Personality Classification App**
            -This app is based on **Supervised Classification** allows you to predict someone's personality as if the person is a 
            **Extrovert**,**Ambivert**,**Introvert**
            -This app includes various visualisations of the data
            -Also includes a model based on `RandomForestClassifier` that will take input from the user and will predict one's personality
            ''')
    st.divider()
    st.header("📶 About the Dataset:")
    st.write("""This synthetic dataset is designed to simulate human personality types —
                Introvert, Extrovert, and Ambivert — based on various behavioral and psychological traits.
                It contains 20,000 entries and 30 columns, including 29 numerical features representing personality
                indicators and 1 label column **(personality_type)**. Class balance: ~33% Introverts, 34% Extroverts, 33% Ambiverts.""")
                
    st.markdown( """
             -`social_energy`:Tendency to gain energy from social interaction \n
                -`alone_time_preference`:Comfort with solitude \n
                -`talkativeness`:Propensity to engage in conversation \n
                -`deep_reflection`:Frequency of deep or introspective thinking \n
                -`group_comfort`:Ease in group environments \n
                -`party_liking`:Enjoyment of parties and social events \n
                -`listening_skill`:Active listening ability \n
                -`empathy`:	Ability to understand other's emotions \n
                -`creativity`:Tendency toward creative thinking \n
                -`organization`:Preference for order, structure, and plans \n
                -`leadership`:	Comfort in leading others \n
                -`risk_taking`:Willingness to take risks \n
                -`public_speaking_comfort`:Comfort level in public speaking situations \n
                -`curiosity`:Interest in learning or exploring \n
                -`routine_preference`:	Preference for routine vs. spontaneity \n
                -`excitement_seeking`:Desire for new and stimulating experiences \n
                -`friendliness`: General social warmth and approachability \n
                -`emotional_stability`: Ability to remain calm and balanced under stress \n
                -`planning`:Tendency to plan ahead \n
                -`spontaneity`:	Acting on impulse or without planning \n
                -`adventurousness`:	Willingness to try new and risky activities \n
                -`reading_habit`:	Frequency of reading books or articles \n
                -`sports_interest`:	Level of interest in sports or physical activities \n
                -`online_social_usage`:	Time spent on social media and online interaction \n
                -`travel_desire`:Interest in travel and exploring new places \n
                -`gadget_usage`:	Frequency of gadget or tech device use \n
                -`work_style_collaborative`:Preference for teamwork vs. solo work \n
                -`decision_speed`:How quickly decisions are made \n
                -`stress_handling`:Ability to manage stress effectively \n

                """)

def plots():
    st.header("Visualisations and plots")
    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(20,14))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='viridis',fmt=".2f",ax=ax)
    st.pyplot(fig)

    st.subheader("Stat Count:")
    feature = st.radio("Pick a feature", df.select_dtypes(include=['int64', 'float64']).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

def predict():
    st.title("🔮 Prediction Time")
    st.divider()
    social_energy=st.slider("Social Energy",0,10,6)
    alone_time_preference=st.slider("Alone Time Preference",0,10,6)
    talkativeness=st.slider("Talkativeness",0,10,6)
    deep_reflection=st.slider("Deep Reflection",0,10,6)
    group_comfort=st.slider("Group Comfort",0,10,6)
    party_liking=st.slider("Party Liking",0,10,6)
    listening_skill=st.slider("Listening Skills",0,10,6)
    empathy=st.slider("Empathy",0,10,6)
    creativity=st.slider("Creativity",0,10,6)
    organization=st.slider("Organization",0,10,6)
    leadership=st.slider("Leadership",0,10,6)
    risk_taking=st.slider("Risk Taking",0,10,6)
    public_speaking_comfort=st.slider("Public Speaking Comfort",0,10,6)
    curiosity=st.slider("Curiosity",0,10,6)
    routine_preference=st.slider("Routine Preference",0,10,6)
    excitement_seeking=st.slider("Excitement Seeking",0,10,6)
    friendliness=st.slider("Friendliness",0,10,6)
    emotional_stability=st.slider("Emotional Stability",0,10,6)
    planning=st.slider("Planning",0,10,6)
    spontaneity=st.slider("Spontaneity",0,10,6)
    adventurousness=st.slider("Adventurousness",0,10,6)
    reading_habit=st.slider("Reading Habit",0,10,6)
    sports_interest=st.slider("Interest in Sports",0,10,6)
    online_social_usage=st.slider("Social Media Usage",0,10,6)
    travel_desire=st.slider("Travel Desire",0,10,6)
    gadget_usage = st.slider("Gadget Usage", 0, 10, 6)
    work_style_collaborative = st.slider("Work Style (Collaborative)", 0, 10, 6)
    decision_speed = st.slider("Decision Speed", 0, 10, 6)
    stress_handling = st.slider("Stress Handling", 0, 10, 6)


    X = [[
    social_energy, alone_time_preference, talkativeness, deep_reflection,
    group_comfort, party_liking, listening_skill, empathy, creativity, organization,
    leadership, risk_taking, public_speaking_comfort, curiosity, routine_preference,
    excitement_seeking, friendliness, emotional_stability, planning, spontaneity,
    adventurousness, reading_habit, sports_interest, online_social_usage,
    travel_desire, gadget_usage, work_style_collaborative, decision_speed,
    stress_handling
    ]]
    if st.button("Run Model"):
        res = model.predict(X)
        if res[0] == 0:
            st.info("🧊 Likely an Introvert")
        elif res[0] == 1:
            st.info("🌀 Likely an Ambivert")
        else:
            st.info("🔥 Likely an Extrovert")

st.divider()



features=df[['social_energy', 'alone_time_preference', 'talkativeness', 'deep_reflection', 'group_comfort', 'party_liking', 'listening_skill', 'empathy', 'creativity', 'organization', 'leadership', 'risk_taking', 'public_speaking_comfort', 'curiosity', 'routine_preference', 'excitement_seeking', 'friendliness', 'emotional_stability', 'planning', 'spontaneity', 'adventurousness', 'reading_habit', 'sports_interest', 'online_social_usage', 'travel_desire', 'gadget_usage', 'work_style_collaborative', 'decision_speed', 'stress_handling']
]
target=df['personality_type']
y_pred = cross_val_predict(model, features, target, cv=5)

def report():
    st.title("Performance Analysis Of Model")
    st.header("📋 Classification Report")
    clfrt = classification_report(target, y_pred)
    st.code(clfrt, language='text')
    st.divider()
    st.header("🎯 Accuracy Score")
    acc=accuracy_score(target,y_pred)
    st.write(f"Accuracy:{acc*100:.2f}%")
    st.header("😵 Confusion Matrix")
    cm = confusion_matrix(target, y_pred)
    st.write(cm)
page = st.sidebar.radio("📍 Navigate", ["Home", "Visualisations", "Predict", "Model Performance"])

if page == "Home":
    about()
elif page == "Visualisations":
    plots()
elif page == "Predict":
    predict()
elif page == "Model Performance":
    report()



