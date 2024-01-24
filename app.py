import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Transport classification model")


file = st.file_uploader("Upload image", type=['png', 'jpeg','gif', 'svg'])
if file: 
    st.image(file)

#PIL convert

img = PILImage.create(file)

#Model
model = load_learner('transport_model.pkl')

# prediction

prediction, pred_id, probs = model.predict(img)
st.success(f"Prediction: {prediction}")
st.info(f"Probiblity: {probs[pred_id]*100:.1f}%")

fig = px.bar(x=probs*100, y=model.dls.vocab)
st.plotly_chart(fig)
