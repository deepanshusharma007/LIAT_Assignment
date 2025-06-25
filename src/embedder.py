import torch
import torchreid
import cv2
import numpy as np
from PIL import Image

class ReIDEmbedder:
    def __init__(self, model_name='osnet_x0_5', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        )
        self.model.eval().to(self.device)

        self.transforms = torchreid.data.transforms.build_transforms(
            height=256,
            width=128,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225]
        )[1]

    def extract(self, image):
        image = cv2.resize(image, (128, 256))
        image = Image.fromarray(image)
        image = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image)
        if isinstance(features, tuple):
            features = features[0]  # take y only if (y, v)
        return features[0].cpu().numpy()