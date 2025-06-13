import streamlit as st
import os
from main import main  # your existing logic in a function

st.title("🎾 Tennis Match Analysis System")

uploaded_video = st.file_uploader("Upload a Tennis Match Video", type=["mp4"])
if uploaded_video:
    with open("input_videos/input_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    if st.button("Start Analysis"):
        st.text("Processing...")
        main()  # runs your existing video analysis logic
        st.video("output_videos/output_video.avi")
