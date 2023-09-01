import base64
import os
import io
import gradio as gr
import requests, json
# from transformers import pipeline
from googletrans import Translator
from dotenv import load_dotenv, find_dotenv
# Load environment variables from .env file
_ = load_dotenv(find_dotenv())
hf_api_key= os.environ['HF_API_KEY']
endpoint_URL= os.environ['HF_API_ICAPTION_BASE']

# Set up the Gradio interface
gr.close_all()

# Load HF_API_KEY from environment variables
# hf_api_key = os.environ['HF_API_KEY']

# Create a pipeline for image captioning
# caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def get_completion(inputs, parameters=None,ENDPOINT_URL=endpoint_URL): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))
# Helper functions
def translate_to_spanish(text):
    translator = Translator()
    translated_text =  translator.translate(text, src='en', dest='es')
    return translated_text.text
 
#Helper function to format the image
def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr= byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))
# Gradio interface functions
def captioner(image):
    base64_image = image_to_base64_str(image)
    caption = get_completion(base64_image)
    original_caption = caption[0]['generated_text']
    translated_caption = translate_to_spanish(original_caption)
    return [original_caption,translated_caption]

iface = gr.Interface(
    fn=captioner,
    inputs=[gr.Image(label="Upload image", type="pil")],
    outputs=[
        gr.Textbox(label="Original Caption"),
        gr.Textbox(label="Spanish Caption")],
    title="Image Captioning with BLIP",
    description="Caption any image using the BLIP model",
    allow_flagging="never",
    examples=[["christmas_dog.jpeg"], ["bird_flight.jpeg"], ["cow.jpeg"]]
)

if __name__ == "__main__":
    iface.launch()