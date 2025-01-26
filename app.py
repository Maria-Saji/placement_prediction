import os
import pickle

import keras
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.set_page_config(page_title="Placement Prediction", page_icon="🎓", layout="centered")

nn = st.sidebar.toggle("Use Neural Network Model", value=True, help="Toggle to use NN trained model, otherwise a random forest classifier")

st.markdown(
    """
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .block-container {
        max-width: 800px;
        padding: 2rem;
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(nn) -> keras.Model | RandomForestClassifier:
    if nn:
        model = keras.models.load_model("./placement_model.keras")
    else:
        with open("./random_forest_clf.pkl", "rb") as fp:
            model = pickle.load(fp)

    assert isinstance(model, keras.Model) or isinstance(model, RandomForestClassifier)
    return model


def predict():
    X = pd.DataFrame(
        [[interns, projects, certifications, extracurricular, placement_training, cgpa, aptitude_score, soft_skills, ssc_marks, hsc_marks]],
        columns=np.array(
            [
                "Internships",
                "Projects",
                "Workshops/Certifications",
                "ExtracurricularActivities",
                "PlacementTraining",
                "CGPA",
                "AptitudeTestScore",
                "SoftSkillsRating",
                "SSC_Marks",
                "HSC_Marks",
            ]
        ),
    )
    model = load_model(nn)
    pred = model.predict(X)
    return pred


st.title("🎓 College Placement Prediction")
st.markdown("### Let's predict your placement chances!")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input("📚 CGPA", 0.0, 10.0, step=0.01)
    interns = st.number_input("💼 No. of internships", 0, 10, step=1)
    projects = st.number_input("🛠️ No. of projects", 0, 10, step=1)
    certifications = st.number_input("📜 No. of certifications", 0, 10, step=1)
    aptitude_score = st.number_input("🎯 Aptitude Test Score", 0.0, 100.0, step=0.01)

with col2:
    soft_skills = st.number_input("🗣️ Soft Skills Rating", 0.0, 5.0, step=0.01)
    extracurricular = st.checkbox("🏃‍♂️ Participated in Extracurricular Activities")
    placement_training = st.checkbox("📋 Completed Placement Training")
    ssc_marks = st.number_input("📊 SSC Marks (%)", 0.0, 100.0, step=0.01)
    hsc_marks = st.number_input("📈 HSC Marks (%)", 0.0, 100.0, step=0.01)

st.markdown("---")

if st.button("✨ Predict My Placement! ✨", type="primary"):
    with st.spinner("Analyzing your profile..."):
        prediction = predict()
        percentage = 0
        pred = False
        if nn:
            percentage = float(prediction[0][0] * 100)
        else:
            pred = prediction[0]

    if nn:
        if prediction > 0.75:
            st.success(f"""
            ### 🎉 Excellent News!
            You have a **{percentage:.1f}%** chance of placement

            You're well-prepared and definitely on track for placement. Keep up the great work!
            """)
        elif prediction > 0.5:
            st.warning(f"""
            ### 📈 Good Prospects
            You have a **{percentage:.1f}%** chance of placement

            You have a good chance, but there's room for improvement. Focus on strengthening your weak areas.
            """)
        else:
            st.error(f"""
            ### 💪 More Effort Needed
            You have a **{percentage:.1f}%** chance of placement

            Don't be discouraged! Focus on improving your skills, particularly in areas like internships, projects, and academics.
            """)

    else:
        if pred:
            st.success("Yes")
        else:
            st.error("NO")
