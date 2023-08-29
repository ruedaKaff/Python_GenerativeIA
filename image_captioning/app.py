#import os
import io
#import IPython.display
from PIL import Image
import base64 
import gradio as gr
from transformers import pipeline

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
gr.close_all()

#hf_api_key = os.environ['HF_API_KEY']

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def summarize(input):
    output = get_completion(input)
    return output[0]['generated_text']

get_completion = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")



# Helper functions
"""
import requests, json

#Image-to-text endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_ITT_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))
"""

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = summarize(base64_image)
    return result #[0]['generated_text']


iface = gr.Interface(fn=captioner,
                     inputs=[gr.Image(label="Upload image", type="pil")],
                     outputs=[gr.Textbox(label="Caption")],
                     title="Image Captioning with BLIP",
                     description="Caption any image using the BLIP model",
                     allow_flagging="never",
                     examples=[["christmas_dog.jpeg"], ["bird_flight.jpeg"], ["cow.jpeg"]])

if __name__ == "__main__":
    iface.launch()