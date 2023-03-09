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


