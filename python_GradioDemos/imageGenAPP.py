import os
import io
import base64
import time
import gradio as gr
import requests, json
from PIL import Image
# import imghdr
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #Readlocal .env file
hf_api_key= os.environ['HF_API_KEY']
endpoint_URL= os.environ['HF_API_SDIFFUSION_BASE']
#Service function


def get_completion(inputs, parameters=None, ENDPOINT_URL=endpoint_URL):
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
        return response.content  # Return binary content
    else:
        return None  # Handle error cases appropriately


def generator (prompt):
    unique_prompt = f"{prompt}_{int(time.time())}"
    output = get_completion(unique_prompt)
    encoded_data = base64.b64encode(output).decode("utf-8")
    result_image = Image.open(io.BytesIO(base64.b64decode(encoded_data)))
    return result_image

demo = gr.Interface (fn= generator,
                     inputs= gr.Textbox(label="Prompt you text"),
                     outputs= gr.Image(label="Image generated"),
                     title="Text to image using StableDiffusion",
                     allow_flagging="never",
                    #  cache=0,
                     examples=["A astronaut on suburbs with a piano","A mecha wizard with lightings in a favela"]
)
demo.launch(server_port=int(os.environ['PORT1']))