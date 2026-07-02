import streamlit as st
import joblib
import numpy as np
import base64

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")


# ---- Load your local image and convert it for the background ----
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


img_base64 = get_base64_image("lily.jpg")
IMAGE_URL = f"data:image/jpeg;base64,{img_base64}"

# ---- Dark Bloom CSS ----
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
}}

.stApp {{
    background-color: #0a0705;
    position: relative;
    overflow: hidden;
}}

.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url("{IMAGE_URL}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    filter: brightness(0.6);
    z-index: 0;
}}

.stApp > header, .stApp > div {{
    position: relative;
    z-index: 1;
}}

h1 {{
    text-align: center;
    color: #ffe3ea;
    font-weight: 700 !important;
    font-size: 2.6em !important;
    text-shadow: 0 0 20px rgba(216, 27, 96, 0.6);
}}

.stMarkdown p {{
    text-align: center;
    color: #e8b8c4;
    font-size: 1.1em;
    font-style: italic;
    font-weight: 400;
}}

div[data-testid="stSlider"] {{
    background: rgba(20, 12, 14, 0.6);
    padding: 15px 20px;
    border-radius: 18px;
    margin-bottom: 15px;
    border: 1px solid rgba(216, 27, 96, 0.3);
    backdrop-filter: blur(6px);
}}

div[data-testid="stSlider"] label {{
    color: #f4c9d3 !important;
}}

.stButton > button {{
    background: linear-gradient(90deg, #ad1457, #d81b60);
    color: #ffe3ea;
    border: none;
    border-radius: 30px;
    padding: 12px 40px;
    font-size: 1.05em;
    font-weight: 600;
    box-shadow: 0 0 20px rgba(216, 27, 96, 0.5);
    transition: transform 0.2s;
    display: block;
    margin: 20px auto;
}}

.stButton > button:hover {{
    transform: scale(1.04);
    box-shadow: 0 0 28px rgba(216, 27, 96, 0.75);
}}

div[data-testid="stAlert"] {{
    background: rgba(20, 12, 14, 0.75);
    border-radius: 18px;
    border: 2px solid #d81b60;
    text-align: center;
    font-size: 1.25em;
    color: #ffe3ea;
    backdrop-filter: blur(6px);
}}

.stMarkdown, p, span, label {{
    color: #f4c9d3;
}}

div[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, #ad1457, #d81b60);
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---- App content ----
model = joblib.load("iris_model.pkl")
species_names = ["Setosa", "Versicolor", "Virginica"]
species_emoji = ["🌷", "🌼", "🌺"]

st.title("🌸 Iris Flower Classifier")
st.markdown("*a quiet little corner to figure out which flower you are* 🌙")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("🌿 Sepal length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("🌿 Sepal width (cm)", 2.0, 4.5, 3.4)
with col2:
    petal_length = st.slider("🌸 Petal length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("🌸 Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("🌙 Predict Species 🌙"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.success(
        f"{species_emoji[prediction]} It's a **{species_names[prediction]}**! {species_emoji[prediction]}"
    )

    st.markdown("**Confidence breakdown:**")
    for name, prob, emoji in zip(species_names, probability, species_emoji):
        st.write(f"{emoji} {name}")
        st.progress(float(prob))
