from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Cargar imagen
image = Image.open(requests.get("https://huggingface.co/datasets/mishig/sample_images/resolve/main/bus.png", stream=True).raw)

# Cargar modelo y procesador
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Prompt multimodal
question = "¿Qué está ocurriendo en esta imagen?"

# Preprocesar
inputs = processor(image, question, return_tensors="pt").to("cuda", torch.float16)

# Generar
out = model.generate(**inputs, max_new_tokens=50)

# Decodificar
print(processor.decode(out[0], skip_special_tokens=True))
