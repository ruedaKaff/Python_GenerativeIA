import os
import io
import gradio as gr
from transformers import pipeline
from googletrans import Translator
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Set up the Gradio interface
gr.close_all()

# Load HF_API_KEY from environment variables
hf_api_key = os.environ['HF_API_KEY']

# Create a pipeline for image captioning
caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

#Helper functions
def translate_to_spanish(text):
    translator = Translator()
    translated_text =  translator.translate(text, src='en', dest='es')
    return translated_text.text

# Gradio interface functions
def captioner(image):
    caption = caption_pipeline(image)
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