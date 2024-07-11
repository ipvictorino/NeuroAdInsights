import streamlit as st
# from langchain.workflow import run_workflow

st.title("🦜🔗 - Neuromarketing Insights")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
uploaded_heatmap = st.file_uploader("Upload corresponding attention heatmap", type=["jpg", "png"])

# if uploaded_image and uploaded_heatmap:
#     # Process the inputs and run the workflow
#     result = run_workflow(uploaded_image, uploaded_heatmap)
#     st.json(result)