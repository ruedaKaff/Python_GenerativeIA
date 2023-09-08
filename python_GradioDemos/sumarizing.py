import os 
import gradio as gr
import requests, json
from dotenv import load_dotenv, find_dotenv
# from transformers import pipeline
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #Readlocal .env file
hf_api_key= os.environ['HF_API_KEY']
endpoint_URL= os.environ['HF_API_SUMMARY_BASE']

# get_completion2 = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

#Helper function Sumarization endpoint

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


#Doing it all at once
# def caption_and_generate(image):
#     caption = captioner(image)
#     image = generate(caption)
#     return [caption, image]

# with gr.Blocks() as demo:
#     gr.Markdown("# Describe-and-Generate game 🖍️")
#     image_upload = gr.Image(label="Your first image",type="pil")
#     btn_all = gr.Button("Caption and generate")
#     caption = gr.Textbox(label="Generated caption")
#     image_output = gr.Image(label="Generated Image")

#     btn_all.click(fn=caption_and_generate, inputs=[image_upload], outputs=[caption, image_output]

def sumarize(input): 
    output = get_completion(input)
    return output[0] ['summary_text']

gr.close_all()
smface = gr.Interface(fn= sumarize, 
                          inputs=gr.Textbox(label="Text to sumarize,", lines=6),
                          outputs=gr.Textbox(label="Result",lines=3),
                          title="Text summarizing DEMO")
smface.launch(server_port=int(os.environ['PORT1']))

