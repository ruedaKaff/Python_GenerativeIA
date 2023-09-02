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


def generator (prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    unique_prompt = f"{prompt}_{int(time.time())}"
    output = get_completion(unique_prompt, params)
    encoded_data = base64.b64encode(output).decode("utf-8")
    result_image = Image.open(io.BytesIO(base64.b64decode(encoded_data)))
    return result_image

demo = gr.Interface (fn= generator,
                     inputs=[
                        gr.Textbox(label="Your prompt"),
                        gr.Textbox(label="Negative prompt"),
                        gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                                 info="In how many steps will the denoiser denoise the image?"),
                        gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7, 
                                  info="Controls how much the text prompt influences the result"),
                        gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512),
                        gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512),
                     ],
                    outputs=[gr.Image(label="Result")],
                    title="Image Generation with Stable Diffusion",
                    description="Generate any image with Stable Diffusion",
                    allow_flagging="never",
                    examples=["A astronaut on suburbs with a piano","A mecha wizard with lightings in a favela"]
                    )
demo.launch(server_port=int(os.environ['PORT1']))