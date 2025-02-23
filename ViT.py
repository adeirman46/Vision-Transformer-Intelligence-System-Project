# import libraries
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets    
# import image
from PIL import Image
from going_modular import data_setup
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()

        # create a layer to convert image into patches
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        self.patch_size = patch_size

        # create a layer to flatten the patch feature maps into single dimension
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)
        
    def forward(self, x):
        # create assertion to check correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0,  f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # perform forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched).permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
        return x_flattened

class ViT(nn.Module):
    def __init__(self, img_size=224, 
                 patch_size=16,
                 num_channels=3,
                 embedding_dim=768,
                 dropout=0.1,
                 mlp_size=3072,
                 num_transformer_layers=12,
                 num_heads=12,
                 num_classes=1000):
        super().__init__()

        # Assert image size is divisible by patch size
        assert img_size % patch_size == 0, "Image must be divisible by patch size"

    
        # 1. Patch Embedding
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        # 2. Create class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)
        
        # 3. Create positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

        # 4. Create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)

        # 5. Create stack of transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                    nhead=num_heads,
                                                                    dim_feedforward=mlp_size,
                                                                    activation='gelu',
                                                                    norm_first=True,
                                                                    batch_first=True),
                                                         num_layers=num_transformer_layers)

        # 6. Create mlp head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        # Get some dimension from x
        batch_size = x.shape[0]

        # Create the patch embedding
        x = self.patch_embedding(x)

        # Expand the class token to be the same size as the batch
        class_token = self.class_token.expand(batch_size, -1, -1)

        # Prepend the class token to the patch embedding
        x = torch.cat([class_token, x], dim=1)

        # Add the positional embedding
        x += self.positional_embedding

        # Pass through the embedding dropout
        x = self.embedding_dropout(x)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Pass 0th index of x through the mlp head
        x = self.mlp_head(x[:, 0])

        return x


