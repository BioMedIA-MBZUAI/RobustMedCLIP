import torch
import torch.nn as nn
import numpy as np
from typing import List
import os
import medclip
from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPVisionModelViT, MedCLIPProcessor
from transformers import AutoTokenizer
from torchvision import transforms
import open_clip
import clip

# Set tokenizers parallelism to prevent the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BACKBONES = {

    'clip': {
        'vit': "ViT-B/16",
        'resnet': "RN50"
    },

    'medclip': {
         "resnet": "MedCLIPVisionModel",
         "vit": "MedCLIPVisionModelViT"
    },

    'biomedclip':{
        'vit': "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    },

    'unimedclip':{
        'vit-B-16': "ViT-B-16-quickgelu",
        # 'vit-L-14': "ViT-L-14-336-quickgelu",
    }
    
}

class BaseZeroShotModel(nn.Module):
    def __init__(self, vision_cls: str, device: str = "cuda"):
        super().__init__()
        self.vision_cls = vision_cls
        self.device = device
        # self.preprocess = transforms.Compose([
        #         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        #     ])
        self.preprocess = None
        self.model = None
        
    
    def text_features(self, input_text: List[str]) -> torch.Tensor:
        inputs = clip.tokenize(input_text, truncate=True).to(self.device)
        # preprocess text
        with torch.no_grad():
            text_features = self.model.encode_text(inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def batch_predict(self, images: torch.Tensor, text_features) -> np.ndarray:
        
        images = self.preprocess(images) if self.preprocess else images
        images = images.to(self.device)
        text_features = text_features.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits_per_image = image_features @ text_features.T

        logits_per_image = torch.nn.functional.softmax(logits_per_image, dim=1)
        # predictions = torch.argmax(logits_per_image, dim=1)
        
        return logits_per_image.cpu()

class ClipZeroShot(BaseZeroShotModel):
    def __init__(self, vision_cls: str, device: str = "cuda"):
        super().__init__(vision_cls, device)
        self.model , _, _ = clip.load(vision_cls, device=device)
        self.preprocess = transforms.Compose([
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.model.to(device)
        self.model.eval()

class MedclipZeroShot(BaseZeroShotModel):
    def __init__(self, vision_cls, device: str = "cuda"):
        super().__init__(vision_cls, device)

        model_name = BACKBONES['medclip'][vision_cls]
        self.model = MedCLIPModel(vision_cls=getattr(medclip, model_name))
        self.model.from_pretrained(input_dir=f"../MedCLIP/pretrained/medclip-{vision_cls}")
        self.model.to(device)
    
    def text_features(self, input_text: List[str]) -> torch.Tensor:
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        # preprocess text
        with torch.no_grad():
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            text_features = self.model.encode_text(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return text_features

class BioMedClipZeroShot(BaseZeroShotModel):
    def __init__(self, vision_cls: str = "a photo of a {}", device: str = "cuda"):
        super().__init__(vision_cls, device)
        self.model , _ = open_clip.create_model_from_pretrained(BACKBONES['biomedclip'][vision_cls])
        self.model.to(device)
        self.model.eval()

    def text_features(self, input_text: List[str]) -> torch.Tensor:
        tokenizer = open_clip.get_tokenizer(BACKBONES['biomedclip'][self.vision_cls])
        inputs = [tokenizer(text).to(next(self.model.parameters()).device, non_blocking=True) for text in input_text]
        inputs = torch.cat(inputs)
        # preprocess text
        with torch.no_grad():
            text_features = self.model.encode_text(inputs)
        return text_features

class UniMedClipZeroShot(BaseZeroShotModel):
    def __init__(self, vision_cls: str = "a photo of a {}", dataset: str = "medmnist", device: str = "cuda"):
        super().__init__(vision_cls, device)
        self.vision_cls = BACKBONES['unimedclip'][vision_cls]
        self.pretrained_weights = "./unimed_clip_vit_b16.pt" if vision_cls == 'vit-B-16' else "./unimed_clip_vit_l14.pt"
        self.text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" if vision_cls == 'vit-B-16' else  "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
        self.model = open_clip.create_model(
                                        self.vision_cls,
                                        self.pretrained_weights,
                                        precision='amp',
                                        device=device,
                                        force_quick_gelu=True,
                                        text_encoder_name=self.text_encoder_name,
                                        )
        self.model.to(device)
        self.model.eval()


    def text_features(self, input_text: List[str]) -> torch.Tensor:
        tokenizer = open_clip.HFTokenizer(self.text_encoder_name, context_length=256)
        inputs = [tokenizer(text).to(next(self.model.parameters()).device, non_blocking=True) for text in input_text]
        inputs = torch.cat(inputs, dim=0)
        # preprocess text
        with torch.no_grad():
            text_features = self.model.encode_text(inputs)
        return text_features



class RobustMedClip(BaseZeroShotModel):
    pass