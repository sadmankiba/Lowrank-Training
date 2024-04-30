from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

# Initialize a feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Load the pre-trained Vision Transformer model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Download an image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/800px-PNG_transparency_demonstration_1.pn:wq
g"
image = Image.open(requests.get(image_url, stream=True).raw)

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Make prediction
outputs = model(**inputs)
logits = outputs.logits

# Retrieve predicted class
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

