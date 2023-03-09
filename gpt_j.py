import gradio as gr
from transformers import pipeline
title = "tory Generator"

# gpt-neo-2.7B   gpt-j-6B

def generate(text,the_model,max_length,temperature,repetition_penalty):
    generator = pipeline('text-generation', model=the_model)
    result = generator(text, num_return_sequences=3,
                       max_length=max_length,
                       temperature=temperature,
                       repetition_penalty = repetition_penalty,
                       no_repeat_ngram_size=2,early_stopping=False)
    return result[0]["generated_text"],result[1]["generated_text"],result[2]["generated_text"]


def complete_with_gpt(text,context,the_model,max_length,temperature,repetition_penalty):
    # Use the last [context] characters of the text as context
    max_length = max_length+context
    return generate(text[-context:],the_model,max_length,temperature,repetition_penalty)

def send(text1,context,text2):
    if len(text1)<context:
        return text1 + text2[len(text1):]
    else:
        return text1 + text2[context:]

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Type here and press enter...", lines=4)
    btn = gr.Button("Generate")
    context = gr.Slider(value=200,label="Truncate input text length (AI's memory)",minimum=1,maximum=500)
    the_model = gr.Dropdown(choices=['gpt2','gpt2-medium','gpt2-large','gpt2-xl','EleutherAI/gpt-neo-2.7B','EleutherAI/gpt-j-6B'],value = 'gpt2',label="Choose model")
    max_length = gr.Slider(value=20,label="Max Generate Length",minimum=1,maximum=50)
    temperature = gr.Slider(value=0.9,label="Temperature",minimum=0.0,maximum=1.0,step=0.05)
    repetition_penalty = gr.Slider(value=1.5,label="Repetition penalty",minimum=0.2,maximum=2,step=0.1)
    output1 = gr.Textbox(lines=4,label='1')
    send1 = gr.Button("Send1 to Origin Textbox").click(send,inputs=[textbox,context,output1],outputs=textbox)
    output2 = gr.Textbox(lines=4,label='2')
    send2 = gr.Button("Send2 to Origin Textbox").click(send,inputs=[textbox,context,output2],outputs=textbox)
    output3 = gr.Textbox(lines=4,label='3')
    send3 = gr.Button("Send3 to Origin Textbox").click(send,inputs=[textbox,context,output3],outputs=textbox)
    btn.click(complete_with_gpt,inputs=[textbox,context,the_model,max_length,temperature,repetition_penalty], outputs=[output1,output2,output3])

demo.launch()