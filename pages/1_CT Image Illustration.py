import streamlit as st
from PIL import Image

imageCT = Image.open('CT_image.png')
st.set_page_config(page_title="CT image Demo")

st.markdown("# CT image illustration")
st.write(' ')
st.image(imageCT)
st.write(' ')