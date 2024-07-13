from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain_openai import OpenAI
import re
import os
import base64
from pathlib import Path
from typing import List, Tuple
path_for_prompt = Path(__file__).parent / "data" / "prompts"

# Reads txt files for prompts
prompt_a1_text = open(f"{path_for_prompt}/prompt_a1.txt", "r").read()
prompt_a2_text = open(f"{path_for_prompt}/prompt_a2.txt", "r").read()
promot_b_text = open(f"{path_for_prompt}/prompt_b.txt", "r").read()
prompt_c_text = open(f"{path_for_prompt}/prompt_c.txt", "r").read()


def process_prompt(prompt_text: str) -> List:
    matches = re.findall(r'<(.*?)>(.*?)</\1>', prompt_text, re.DOTALL)
    matches = {match[0]: match[1].strip() for match in matches}

    system_message = SystemMessage(content=matches.get('role'))
    human_message = HumanMessage(content="\n".join(value for key, value in matches.items() if key != 'role'))

    messages = [
        system_message,
        human_message
    ]
    return messages

class LangChainWorkflow:
    def __init__(self, openai_api_key, deployment_name):
        self.openai_api_key = openai_api_key
        self.deployment_name = deployment_name

    def create_llm(
        self,
        max_tokens: int = 512,
        temperature: float = 0.0,
        openai_api_key: str = '',
        deployment_name: str = '',
    ):
        if not openai_api_key:
            self.openai_api_key = openai_api_key
        if not deployment_name:
            self.deployment_name = deployment_name

        llm = OpenAI(
            api_key=self.openai_api_key,
            deployment_name=self.deployment_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return llm
    
    def process_prompt(self, llm: OpenAI, messages: List, custom_messages: List = None, append_answers: bool = False) -> Tuple:
        if custom_messages:
            messages += custom_messages
        
        try:
            answer = llm.invoke(messages)
        except Exception as e:
            raise Exception(f"An error occurred while processing the prompt: {e}")
        
        if append_answers:
            messages.append(AIMessage(content=answer))
    
    def run(self, image_bytes: bytes, heatmap_bytes: bytes):


        llm = self.create_llm()

        human_message_original_image = HumanMessage(content={
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;{image_bytes}"
            }
        })

        heatmap_bytes = HumanMessage(content={
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;{heatmap_bytes}"
            }
        })

        messages_a1 = process_prompt(prompt_a1_text)
        messages_a2 = process_prompt(prompt_a2_text)
        messages_b = process_prompt(promot_b_text)
        messages_c = process_prompt(prompt_c_text)

        # Read input images

        response = self.process_prompt(llm, messages_a1, human_message_original_image, append_answers=True)
        response_a = self.process_prompt(llm, messages_a2, heatmap_bytes, append_answers=True)
        response_b = self.process_prompt(llm, messages_b)
        response_c = self.process_prompt(llm, messages_c, append_answers=True)
        return response_a, response_b, response_c
        

if __name__ == "__main__":

    image_path = f"{Path(__file__).parent / 'data/images/image1.png'}"
    heatmap_path = f"{Path(__file__).parent / 'data/images/image1_heatmap.jpeg'}"

    with open(image_path, "rb") as image_file:
        image_bytes = base64.b64encode(image_file.read())
    with open(heatmap_path, "rb") as heatmap_file:
        heatmap_bytes = base64.b64encode(heatmap_file.read())

    langchai_handler = LangChainWorkflow(
        openai_api_key=os.getenv("OPENAI_API_KEY"), 
        deployment_name="gpt-4o"
    )

    langchai_handler.run(image_bytes, heatmap_bytes)

