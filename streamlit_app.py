import streamlit as st
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
import os

# Set page configuration
st.set_page_config(page_title="Vision Transformer Inference", layout="wide")

# Get class names from dataset directory
# def get_class_names(dataset_path='dataset'):
#     return sorted(os.listdir(dataset_path))

def get_class_names():
    classes = ['ayam_geprek',
                'bakso',
                'bakwan',
                'bubur_ayam',
                'cireng',
                'ketoprak',
                'klepon',
                'mie_ayam',
                'nasi_kuning',
                'rawon',
                'soto',
                'tahu_sumedang']
    return classes



# Define the model loading function
def load_vit_model(weights_path, num_classes, device='cuda'):
    """Load Vision Transformer model with saved weights"""
    # Create the base model
    model = torchvision.models.vit_b_16(weights='IMAGENET1K_SWAG_E2E_V1')
    
    # Modify the classification head
    embedding_dim = 768
    model.heads = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim, out_features=num_classes)
    )
    
    # Load the state dict
    state_dict = torch.load(weights_path, map_location=device)
    
    # Handle potential key mismatches
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(state_dict)
    
    # Load the weights
    model.load_state_dict(model_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

# Define transforms
def get_transforms():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image, transforms, class_names, device):
    """Make prediction on a single image"""
    image_tensor = transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        
    return class_names[predicted_idx], confidence

def main():
    st.title("Vision Transformer Image Classification")
    st.write("Upload an image to classify it using the Vision Transformer model")
    
    # Get class names from dataset directory
    class_names = get_class_names()
    num_classes = len(class_names)
    
    # Display class names in sidebar
    st.sidebar.title("Model Configuration")
    # st.sidebar.write("Available Classes:")
    # st.sidebar.write(", ".join(class_names))
    
    # Model weights path
    weights_path = st.sidebar.text_input(
        "Model weights path:",
        "pretrained_vit_v4.pth"
    )
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.write(f"Using device: {device}")
    st.sidebar.write(f"Number of classes: {num_classes}")
    
    # Load model when configurations are set
    try:
        model = load_vit_model(
            weights_path=weights_path,
            num_classes=num_classes,
            device=device
        )
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            transforms = get_transforms()
            predicted_class, confidence = predict_image(
                model=model,
                image=image,
                transforms=transforms,
                class_names=class_names,
                device=device
            )
            
            # Display results
            with col2:
                st.write("### Prediction Results")
                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Create a progress bar for confidence
                st.progress(confidence)
                
                # Display top-3 predictions
                with torch.no_grad():
                    output = model(transforms(image).unsqueeze(0).to(device))
                    probabilities = torch.softmax(output, dim=1)[0]
                    top3_prob, top3_idx = torch.topk(probabilities, 3)
                    
                st.write("### Top 3 Predictions")
                for idx, (prob, class_idx) in enumerate(zip(top3_prob, top3_idx)):
                    st.write(f"{idx+1}. {class_names[class_idx]} ({prob.item():.2%})")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()