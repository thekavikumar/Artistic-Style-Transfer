from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

app = FastAPI()

# Allow requests from the frontend
origins = [
    "http://localhost:3000",
    "https://arti-style.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained VGG19 model
vgg = models.vgg19(pretrained=True).features

# Freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# Define image transformation function
def load_image_from_url(image_url, max_size=128, shape=None):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Resize image
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

# Helper function to un-normalize and convert tensor to an image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return Image.fromarray((image * 255).astype('uint8'))

# Function to get features from VGG19
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content representation
                  '28': 'conv5_1'}
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Function to calculate Gram Matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

@app.get("/style-transfer/")
async def style_transfer(content_url, style_url):
    # Load content and style images
    content = load_image_from_url(content_url).to(device)
    style = load_image_from_url(style_url, shape=content.shape[-2:]).to(device)
    
    # Get content and style features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Create target image
    target = content.clone().requires_grad_(True).to(device)
    
    # Define weights for each style layer
    style_weights = {'conv1_1': 1,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}
    
    content_weight = 1  # alpha
    style_weight = 1e3  # beta
    
    # Set optimizer and hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = 500  # Adjust number of iterations for faster processing
    
    for ii in range(1, steps + 1):
        target_features = get_features(target, vgg)
        
        # Calculate content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        # Calculate style loss
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)
        
        # Calculate total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Update target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Convert target tensor to image
    final_img = im_convert(target)
    buf = BytesIO()
    final_img.save(buf, format='JPEG')
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/jpeg")
