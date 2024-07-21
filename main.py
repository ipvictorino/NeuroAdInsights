from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from pathlib import Path
import base64
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)

from orchestrator import LangChainWorkflow

app = FastAPI()

class ImageNames(BaseModel):
    image_name: Optional[str] = None
    heatmap_name: Optional[str] = None

async def resize_image(self, image_data):
    logging.error("Image size is too large. Must be less than 20 MB in size.")
    return image_data  # TODO: Implement resizing logic

async def load_image(self, images_dir, image_path):
    image_file_path = images_dir / image_path
    with open(image_file_path, "rb") as image_file:
        image_data = image_file.read()
        if len(image_data) > 20 * 1024 * 1024:
            image_data = self.resize_image(image_data)
        return base64.b64encode(image_data).decode('utf-8')
    
async def read_image_file(file: UploadFile):
    logging.info(f"Reading image file: {file.filename}...")
    image_data = await file.read()
    if len(image_data) > 20 * 1024 * 1024:
        logging.error("Image size is too large. Must be less than 20 MB in size.")
        # TODO: Implement resizing logic if needed
    return base64.b64encode(image_data).decode('utf-8')

@app.post("/process")
async def process_image(
    image_name: Optional[str] = None,
    heatmap_name: Optional[str] = None,
    image_file: Optional[UploadFile] = File(None),
    heatmap_file: Optional[UploadFile] = File(None),
):

    images_dir = Path(__file__).parent / "data" / "images"
    
    
    if image_name and heatmap_name:
        logging.info(f"Loading image and heatmap from file names")
        try:
            image_base64_str = await load_image( images_dir, image_name)
            heatmap_base64_str = await load_image(images_dir, heatmap_name)
            logging.info(f"Image and heatmap files read successfully")

        except Exception as e:
            raise HTTPException(status_code=404, detail="Image not found")
    elif image_file and heatmap_file:
        logging.info(f"Loading image and heatmap from file uploads")
        try:
            logging.info(f"Received image_file: {image_file.filename}, size: {image_file.size}")
            logging.info(f"Received heatmap_file: {heatmap_file.filename}, size: {heatmap_file.size}")
            image_base64_str = await read_image_file(image_file)
            heatmap_base64_str = await read_image_file(heatmap_file)
            logging.info(f"Image and heatmap files read successfully")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Error processing files")
    else:
        logging.error("Image and heatmap files are required")
        # logs input args
        logging.info(f"{image_name}, {heatmap_name}, {image_file}, {heatmap_file}")
        raise HTTPException(status_code=400, detail="Image and heatmap files are required")
    
    if not image_base64_str or not heatmap_file:
        raise HTTPException(status_code=400, detail="Image and heatmap files are required")
    # Initialize the LangChainWorkflow
    workflow = LangChainWorkflow(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        deployment_name="gpt-4o"
    )
    # return ["answer1", "answer"]
    response_a, response_b, response_c = workflow.run(
        image_base64_str, heatmap_base64_str)
    
    logging.info(f"Processed images.")

    return {
        "response_a": response_a.content,
        "response_b": response_b.content,
        "response_c": response_c.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)