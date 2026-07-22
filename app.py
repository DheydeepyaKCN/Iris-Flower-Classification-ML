import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#1e1b4b,#312e81);
color:white;
}

h1{
text-align:center;
color:white;
font-size:45px;
font-weight:bold;
}

p{
text-align:center;
font-size:18px;
}

.stButton>button{
background:#8b5cf6;
color:white;
border:none;
border-radius:12px;
padding:12px 28px;
font-size:18px;
font-weight:bold;
transition:0.3s;
}

.stButton>button:hover{
background:#7c3aed;
transform:scale(1.05);
}

div[data-testid="stSuccess"]{
background:#14532d;
border-radius:12px;
}

div[data-testid="stInfo"]{
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ---------------- #

st.sidebar.title("🧠 Model Information")
st.sidebar.write("**Algorithm:** K-Nearest Neighbors")
st.sidebar.write("**K Value:** 4")
st.sidebar.write("**Dataset:** Iris Dataset")
st.sidebar.write("**Classes:** 3")
st.sidebar.write("**Framework:** Scikit-learn")

# ---------------- Title ---------------- #

st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the flower measurements below and let the Machine Learning model predict the Iris species."
)

st.divider()

# ---------------- Inputs ---------------- #

sepal_length = st.slider(
    "🌿 Sepal Length (cm)",
    4.0, 8.5, 5.8, 0.1
)

sepal_width = st.slider(
    "🌿 Sepal Width (cm)",
    2.0, 4.5, 3.0, 0.1
)

petal_length = st.slider(
    "🌸 Petal Length (cm)",
    1.0, 7.5, 4.3, 0.1
)

petal_width = st.slider(
    "🌸 Petal Width (cm)",
    0.1, 2.8, 1.3, 0.1
)

# ---------------- Prediction ---------------- #

if st.button("🔍 Predict Flower", use_container_width=True):

    features = np.array([[

        sepal_length,
        sepal_width,
        petal_length,
        petal_width

    ]])

    prediction = model.predict(features)[0]

    confidence = np.max(model.predict_proba(features))*100

    flower = {
        0: "🌼 Iris Setosa",
        1: "🌺 Iris Versicolor",
        2: "🌸 Iris Virginica"
    }

    st.success(f"### Prediction\n\n{flower[prediction]}")

    st.info(f"📊 Confidence: **{confidence:.2f}%**")

st.divider()

st.caption("Built with ❤️ using Python • Scikit-learn • Streamlit")