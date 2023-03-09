from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import gradio as gr
import torch
from PIL import Image

model_id = 'andite/anything-v4.0'
prefix = ''
     
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
  model_id,
  torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  scheduler=scheduler)

pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
  model_id,
  torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  scheduler=scheduler)

if torch.cuda.is_available():
  pipe = pipe.to("cuda")
  pipe_i2i = pipe_i2i.to("cuda")

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""

def inference(prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt="", auto_prefix=False):

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
  prompt = f"{prefix} {prompt}" if auto_prefix else prompt

  try:
    if img is not None:
      return img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator), None
    else:
      return txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator), None
  except Exception as e:
    return None, error_str(e)

def txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator):

    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)
    
    return result.images[0]

def img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator):

    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe_i2i(
        prompt,
        negative_prompt = neg_prompt,
        init_image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        width = width,
        height = height,
        generator = generator)
        
    return result.images[0]

css = """.main-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.main-div div h1{font-weight:900;margin-bottom:7px}.main-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Tab("Stable Diffustion andite/anything-v4.0 model"):
            with gr.Column(scale=55):
                with gr.Group():
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder=f"{prefix} [your prompt]").style(container=False)
                        generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))

                    image_out = gr.Image(height=512)
                error_output = gr.Markdown()

            with gr.Column(scale=45):
                with gr.Tab("Options"):
                    with gr.Group():
                        neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")
                        auto_prefix = gr.Checkbox(label="Prefix styling tokens automatically ()", value=prefix, visible=prefix)

                        with gr.Row():
                            guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                            steps = gr.Slider(label="Steps", value=25, minimum=2, maximum=75, step=1)

                        with gr.Row():
                            width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
                            height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

                        seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

                with gr.Tab("Image to image"):
                    with gr.Group():
                        image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                        strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Tab("LLM models"):
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
        


    auto_prefix.change(lambda x: gr.update(placeholder=f"{prefix} [your prompt]" if x else "[Your prompt]"), inputs=auto_prefix, outputs=prompt, queue=False)
    inputs = [prompt, guidance, steps, width, height, seed, image, strength, neg_prompt, auto_prefix]
    outputs = [image_out, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)



demo.queue(concurrency_count=1)
demo.launch()
