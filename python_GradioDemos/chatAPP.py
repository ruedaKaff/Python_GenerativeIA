import os
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60
from transformers import pipeline

# Load environment variables
_ = load_dotenv(find_dotenv())  # Read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Create a text generation pipeline using Hugging Face Transformers
# text_generator = pipeline("text-generation", model="tiiuae/falcon-40b-instruct", trust_remote_code=True)

# Define the Gradio interface
def generate_text(prompt, slider):
    generated_text='hellow world'
    # generated_text = text_generator(prompt, max_length=100)[0]['generated_text']
    return generated_text

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_text,
    inputs=[gr.Textbox(label="Enter your prompt"),gr.Slider(label="Max new tokens", value=20,  maximum=1024, minimum=1)],
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation with tiiuae/falcon-40b-instruct",
    live=True
)
interface.launch(server_port=int(os.environ['PORT1']))

