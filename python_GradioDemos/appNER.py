import os 
import gradio as gr
import requests, json
# from transformers import pipeline
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #Readlocal .env file
hf_api_key= os.environ['HF_API_KEY']
endpoint_URL= os.environ['HF_API_NER_BASE']

# get_completion2 = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

#Helper function Sumarization endpoint
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity_group'].startswith('I-') and merged_tokens[-1]['entity_group'] == 'B-' + token['entity_group'][2:]:
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

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


def ner(input): 
    output = get_completion(input)
    merged_tokens = merge_tokens(output);
    entities = [{'start': token['start'], 'end': token['end'], 'entity': token['entity_group']} for token in merged_tokens]
    return {"text":input, "entities": entities }

smface =gr.Interface(fn= ner, 
                        inputs=[gr.Textbox(label="Text to find entities,", lines=2)],
                        outputs=[gr.HighlightedText(label="Text with entities")],
                        title="NER DEMO",
                        decription="Find entities using the `dslim/bert-baseNER` model under the hood !",
                        allow_flaggin="never",
                        examples=["My name is Andrew and I live in California", "My name is Poli and work at HuggingFace"])
                     
                        
smface.launch(server_port=int(os.environ['PORT1']))
gr.close_all()
