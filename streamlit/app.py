import streamlit as st
import requests
from streamlit_extras.stylable_container import stylable_container

st.title("🦜🔗 - Neuromarketing Insights")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
uploaded_heatmap = st.file_uploader(
    "Upload corresponding attention heatmap", type=["jpg", "png"])

if st.button("Analyze"):
    if uploaded_image is not None and uploaded_heatmap is not None:
        with st.spinner("Your Ad Insights are being generated. Please hold tight...", ):
            files = {
                "image_file": (uploaded_image.name, uploaded_image, uploaded_image.type),
                "heatmap_file": (uploaded_heatmap.name, uploaded_heatmap, uploaded_heatmap.type)
            }

            response = requests.post(
                "http://localhost:8000/process", files=files)

            if response.status_code == 200:
                result = response.json()

                st.success(body="Success! ", icon="🎉")

                with st.expander("See Results"):

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(uploaded_image, caption="Uploaded Image",
                                 use_column_width=True)
                    with col2:
                        st.image(
                            uploaded_heatmap, caption="Uploaded Heatmap", use_column_width=True)

                    with stylable_container(
                        "codeblock",
                        """
                        code {
                            white-space: pre-wrap !important;
                        }
                        """,
                    ):
                        st.markdown("### Main Insights")
                        st.markdown(f"**Insight A - Saliency Description:**")
                        st.markdown(result['response_a'])
                        st.markdown(f"**Insight B - Cognitive Description:**")
                        st.markdown(result['response_b'])
                        st.markdown(f"**Detailed Summary:**")
                        st.markdown(result['response_c'])
            else:
                st.error("Error processing the files")
    else:
        st.error("Please upload an image and a heatmap file")
