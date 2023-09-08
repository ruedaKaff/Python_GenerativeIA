import os
import io
import base64
import time
import gradio as gr
import requests, json
from PIL import Image
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #Readlocal .env file
hf_api_key= os.environ['HF_API_KEY']
TTI_Endpoint= os.environ['HF_API_SDIFFUSION_BASE']
ITT_Endpoint= os.environ['HF_API_ICAPTION_BASE']

#Service function


def get_completion(inputs, parameters=None, ENDPOINT_URL=""):
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
    if response.status_code == 200:
        # Check if the response content is binary data (e.g., image)
        content_type = response.headers.get("Content-Type", "")
        if "image" in content_type:
            # Decode binary content (e.g., image)
            return response.content
        else:
            return json.loads(response.content.decode("utf-8"))  # Return content as is"  # Handle error cases appropriately

#Bringing the functions from lessons 3 and 4!
def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def captioner(image):
    base64_image = image_to_base64_str(image)
    unique_prompt = f"{base64_image}_{int(time.time())}"
    result = get_completion(unique_prompt, None, ITT_Endpoint)
    print(result)
    return result[0]['generated_text']

def generate(prompt):
    unique_prompt = f"{prompt}_{int(time.time())}"
    output = get_completion(unique_prompt, None, TTI_Endpoint)
    encoded_data = base64.b64encode(output).decode("utf-8")
    result_image = Image.open(io.BytesIO(base64.b64decode(encoded_data)))
    # result_image = base64_to_pil(output)
    return result_image


with gr.Blocks() as demo:
    gr.Markdown("# Describe-and-Generate game 🖍️")
    image_upload = gr.Image(label="Your first image",type="pil")
    with gr.Row():
        with gr.Column(scale=2) :
            caption = gr.Textbox(label="Generated caption, try to be especific whit details if you want to promp it yourselft)")
        with gr.Column(scale=2) :
            btn_caption = gr.Button("Generate caption")
            btn_image = gr.Button("Generate image")
        
    
    
    image_output_style = "max-width: 512px; max-height: 512px; width: auto; height: auto; display: block; margin: auto;"
    image_output = gr.Image(label="Generated Image", style=image_output_style)
    
    btn_caption.click(fn=captioner, inputs=[image_upload], outputs=[caption])
    btn_image.click(fn=generate, inputs=[caption], outputs=[image_output])

gr.close_all()
demo.launch( server_port=int(os.environ['PORT2']))