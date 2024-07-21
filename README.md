# NeuroAdInsights
![In Progress](https://img.shields.io/badge/Status-In%20Progress-blue)

![ChatGPT](https://img.shields.io/badge/OpenAI-%23412991?logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%235F4687?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-%2300C7B7?logo=fastapi&logoColor=white)
![Langchain](https://img.shields.io/badge/Langchain-%23FFD166?logo=langchain&logoColor=white)

## Overview :book:
NeuroAdInsights is a neuromarketing tool designed to generate insights into the visual salience and cognitive load of marketing assets using Large Language Models (LLMs). By processing images and attention heatmaps, this tool provides detailed analyses that can help optimize marketing strategies.

- [NeuroAdInsights](#neuroadinsights)
  - [Overview :book:](#overview-book)
  - [Features :sparkles:](#features-sparkles)
  - [Installation :hammer\_and\_wrench:](#installation-hammer_and_wrench)
  - [Usage :rocket:](#usage-rocket)
  - [Project Structure :file\_folder:](#project-structure-file_folder)
  - [Contact :mailbox\_with\_mail:](#contact-mailbox_with_mail)


## Features :sparkles:
- **Multi-modal Workflow**: Integrates image and heatmap data to generate comprehensive insights.
- **LLM Integration**: Utilizes advanced language models to interpret and summarize visual data.
- **Streamlit Interface**: User-friendly web interface for uploading images and heatmaps and viewing results.

## Installation :hammer_and_wrench:
1. Clone the repository:
    ```sh
    git clone git@github.com:ipvictorino/NeuroAdInsights.git
    cd NeuroAdInsights
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key:
3.1. Create an OpenAI Account
    Create an account on [OpenAI](https://platform.openai.com/).

3.2. Get your API key
    Generate an API key on the API keys [page](https://platform.openai.com/account/api-keys) and store it securely.

3.3. Set up Environment Variable:
    ```sh
    export OPENAI_API_KEY='your_openai_api_key'
    ```

## Usage :rocket:
1. Start the backend API 
    ```sh
    uvicorn main:app --reload
    ```
2. Run the Streamlit app:
    ```sh
    streamlit run streamlit/app.py
    ```
3. A web browser will open automatically with the Streamlit app. Otherwise, navigate to `http://localhost:8501` to access the app.

4. Upload an image and heatmap to generate insights.

## Project Structure :file_folder:

- `main.py`: Contains the FastAPI backend logic for processing images and heatmaps.
- `app.py`: Streamlit application for user interaction.
- `orchestrator.py`: Contains the core logic for processing prompts and interacting with the LLM.
- `data/prompts`: Directory containing prompt templates.
- `data/images`: Directory for storing uploaded images and heatmaps.


## Contact :mailbox_with_mail:
For any questions, please contact [invfi1997@gmail.com](mailto:invfi1997@gmail.com).