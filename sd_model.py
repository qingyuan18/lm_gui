from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import boto3

s3_resource = boto3.resource('s3')
sm_endpoint_1 = 'AIGC-Quick-Kit-8f46c6b9-be46-48a0-b7b6-6c01dacedcd6'

def get_bucket_and_key(s3uri):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]
    return bucket, key


def get_images(response):
    imags=[]
    try:
        predictions = response['result']
        print(predictions)
        for prediction in predictions:
            bucket, key = get_bucket_and_key(prediction)
            obj = s3_resource.Object(bucket, key)
            bytes = obj.get()['Body'].read()
            image = Image.open(io.BytesIO(bytes))
            imags.append(image)
    except Exception as e:
        traceback.print_exc()
        print(e)
    print(f"Time taken: {time.time() - start}s")
    return images

def sm_sd_inference(sm_endpoint=sm_endpoint_1, 
                    prompt, guidance, steps, width=512, 
                    height=512, seed=0, img=None, 
                    strength=0.5, neg_prompt="", 
                    auto_prefix=False, generator, 
                    inf_type='text_to_image'):
    payload = json.dumps(test_data.tolist())
    sagemaker = boto3.client('sagemaker-runtime')
    response = sagemaker.invoke_endpoint(
        EndpointName=sm_endpoint,
        ContentType='application/json',
        Body=payload
    )
    result = response['Body'].read()
    result = json.loads(result)
    get_images(result)


model_id = 'andite/anything-v4.0'
prefix = ''
sm_endpoint='AIGC-Quick-Kit-8f46c6b9-be46-48a0-b7b6-6c01dacedcd6'
     
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

    
def sm_inference(sm_endpoint, prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt="", auto_prefix=False):
    pass
    
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


