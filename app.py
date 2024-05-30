
from flask import Flask, request, send_file
import torch
from PIL import Image
from diffusers.utils import make_image_grid
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

app = Flask(__name__)

# Load the models
img2img_model = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, low_cpu_mem_usage=True)
img2img_model = img2img_model.to("cpu")

text2img_model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",torch_dtype=torch.float16, low_cpu_mem_usage=True)
text2img_model = text2img_model.to("cpu")

@app.route('/generate_image', methods=['POST'])
def generate_image():
    prompt = request.form['prompt']
    strength = float(request.form.get('strength', 0.8))

    if 'image' in request.files:
        image_file = request.files['image']
        init_image = Image.open(image_file).convert("RGB")
        init_image = init_image.resize((512, 512))

        with torch.no_grad():
            output_image = img2img_model(prompt, image=init_image, strength=strength).images[0]

        # Create an image grid
        grid = make_image_grid([init_image, output_image], rows=1, cols=2)
        output_path = "output_grid.jpeg"
        grid.save(output_path)

        return send_file(output_path, mimetype='image/jpeg')
    else:
        with torch.no_grad():
            image = text2img_model(prompt).images[0]

        save_path = "generated_image.png"
        image.save(save_path)
        return send_file(save_path, mimetype='image/png')

if __name__ == '__main__':
    app.run()
