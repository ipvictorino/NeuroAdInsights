from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain_openai import ChatOpenAI
import re
import os
import base64
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union
import logging

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING)

logger = logging.getLogger("LangChainWorkflow")
logger.setLevel(logging.DEBUG)

logging.getLogger("httpx").setLevel(logging.WARNING)

# Path to prompts
path_for_prompt = Path(__file__).parent / "data" / "prompts"

# Reads txt files for prompts
prompt_a1_text = open(f"{path_for_prompt}/prompt_a1.txt", "r").read()
prompt_a2_text = open(f"{path_for_prompt}/prompt_a2.txt", "r").read()
prompt_b_text = open(f"{path_for_prompt}/prompt_b.txt", "r").read()
prompt_c_text = open(f"{path_for_prompt}/prompt_c.txt", "r").read()


def process_prompt(prompt_text: str) -> Union[SystemMessage, HumanMessage]:
    matches = re.findall(r'<(.*?)>(.*?)</\1>', prompt_text, re.DOTALL)
    matches = {match[0]: match[1].strip() for match in matches}

    human_message = HumanMessage(content=[{"type": "text", "text": "\n".join(
        value for key, value in matches.items() if key != 'role')}])

    if 'role' in matches:
        system_message = SystemMessage(content=matches.get('role'))

        messages = [
            system_message,
            human_message
        ]
    else:
        messages = [human_message]

    return messages


class LangChainWorkflow:
    def __init__(self, openai_api_key: str, deployment_name: str):
        logger.info("Initialising LangChainWorkflow...")
        self.openai_api_key = openai_api_key
        self.deployment_name = deployment_name

    def create_llm(
        self,
        max_tokens: int = 512,
        temperature: float = 0.7,
        openai_api_key: str = '',
        deployment_name: str = '',
    ) -> ChatOpenAI:
        
        logger.debug("Creating LLM...")
        if not openai_api_key:
            openai_api_key = self.openai_api_key
        if not deployment_name:
            deployment_name = self.deployment_name

        llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name=deployment_name,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Test llm
        try:
            llm.invoke("0")
            logger.debug("LLM created successfully.")
        except Exception as e:
            raise Exception(f"An error occurred while creating the LLM: {e}")
        return llm

    def process_prompt(self, llm: ChatOpenAI, messages: List, append_answers: bool = False) -> AIMessage:

        try:
            answer = llm.invoke(messages)
        except Exception as e:
            raise Exception(
                f"An error occurred while processing the prompt: {e}")

        if append_answers:
            messages.append(answer)

        return answer

    def run(self, image_base64_str: str, heatmap_base64_str: str) -> Tuple[AIMessage, AIMessage, AIMessage]:

        ########################################################
        # Step 1: Create LLM
        ########################################################

        llm = self.create_llm()

        # TODO: Automatically define image extension png jpeg etc
        original_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64_str}"
            }
        }

        heatmap_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{heatmap_base64_str}"
            }
        }

        ########################################################
        # Step 2: Process Prompts and create list of messages
        ########################################################

        logger.info("Processing prompts for tasks A1, A2, B, and C...")
        messages_a1_original = process_prompt(prompt_a1_text)
        messages_a2_original = process_prompt(prompt_a2_text)
        messages_b_original = process_prompt(prompt_b_text)
        messages_c = process_prompt(prompt_c_text)

        # Append original image to HumanMessage within messages_a1 and heatmap_image to HumanMessage within messages_a2
        messages_a1 = deepcopy(messages_a1_original)
        messages_a2 = deepcopy(messages_a2_original)
        messages_b = deepcopy(messages_b_original)
        messages_a1[-1].content.append(original_image)
        messages_a2[-1].content.append(heatmap_image)
        messages_b[-1].content.append(original_image)
        logger.info("Prompts processed and images appended successfully")

        ########################################################
        # Step 3: Process prompts using LLM 
        ########################################################

        # 3.1 Chain - Provide image and ask model to describe key elements of advert + Provide corresponding attention heatmap and ask model to describe the most visually salient elements based on the heatmap.
        logger.info("Running Task A - Chain of thought...")
        logger.debug("\tTask A1 - Describe key elements of advert (description and purpose)")
        response = self.process_prompt(llm, messages_a1)
        logger.debug("\tTask A2 - Describe visually salient elements based on heatmap.")
        response_a = self.process_prompt(
            llm, messages_a1_original + [response] + messages_a2, append_answers=True)


        # 3.2 Provide image and ask model to assess perceptual/cognitive load of the asset.
        logger.info("Running Task B - Cognitive load...")
        response_b = self.process_prompt(llm, messages_b)

        # 3.3. Feed output from Processes A & B and ask model to summarise the information and ensure that the final output has the desired JSON format.
        logger.info("Running Task C - Summarise information...")
        response_c = self.process_prompt(
            llm, messages_c + [response_a, response_b], append_answers=True)
        
        # TODO Parallelize calls A and B using chain
        logger.info("Tasks A, B and C completed successfully.")
        
        return response_a, response_b, response_c


def resize_image(image_data):
    logger.error("Image size is too large. Must be less than 20 MB in size.")
    return image_data  # TODO: Implement resizing logic


def load_image(images_path):
    with open(images_path, "rb") as image_file:
        image_data = image_file.read()
        if len(image_data) > 20 * 1024 * 1024:
            image_data = resize_image(image_data)
        return base64.b64encode(image_data).decode('utf-8')


if __name__ == "__main__":

    images_dir = Path(__file__).parent / "data" / "images"
    image_name = 'image1.png'
    heatmap_name = 'image1_heatmap.jpeg'

    image_base64_str = load_image(images_dir / image_name)
    heatmap_base64_str = load_image(images_dir / heatmap_name)

    langchai_handler = LangChainWorkflow(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        deployment_name="gpt-4o"
    )

    response_a, response_b, response_c = langchai_handler.run(
        image_base64_str, heatmap_base64_str)

    logger.info(f"- Response A: {response_a.content}")
    logger.info(f"- Response B: {response_b.content}")
    logger.info(f"- Response C: {response_c.content}")
