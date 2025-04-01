import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import os
import medclip
from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPVisionModelViT, MedCLIPProcessor
from transformers import AutoTokenizer
from torchvision import transforms
import open_clip
import clip

from timm.models.vision_transformer import VisionTransformer as timm_ViT
from utils import LoRA_ViT_timm, LoRA_resnet, _MODELS


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
        'vit-L-14': "ViT-L-14-336-quickgelu",
    },
    'rmedclip':{
        'vit': "path/to/robustmedclip/model",
        'resnet': "path/to/robustmedclip/model",
    },
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

        logits_per_image = torch.nn.functional.softmax(logits_per_image.float(), dim=1)
        # predictions = torch.argmax(logits_per_image, dim=1)
        
        return logits_per_image.cpu()

class ClipZeroShot(BaseZeroShotModel):
    def __init__(self, vision_cls: str, device: str = "cuda", **kwargs):
        super().__init__(vision_cls, device)
        self.model , _ = clip.load(BACKBONES['clip'][vision_cls], device=device)
        self.preprocess = transforms.Compose([
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.model.to(device)
        self.model.eval()

class MedclipZeroShot(BaseZeroShotModel):
    def __init__(self, vision_cls, device: str = "cuda", **kwargs):
        super().__init__(vision_cls, device)

        model_name = _MODELS['medclip']['backbones'][self.vision_cls]
        self.text_encoder_name = _MODELS['medclip']['text_encoder_name'][self.vision_cls]
        self.model = MedCLIPModel(vision_cls=eval(model_name))
        self.model.from_pretrained(input_dir=_MODELS['medclip']['pretrained_weights'][self.vision_cls])
        self.model.to(device)
    
    def text_features(self, input_text: List[str]) -> torch.Tensor:
        tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        # preprocess text
        with torch.no_grad():
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            text_features = self.model.encode_text(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return text_features

class BioMedClipZeroShot(BaseZeroShotModel):
    def __init__(self, vision_cls: str = "a photo of a {}", device: str = "cuda", **kwargs):
        super().__init__(vision_cls, device)
        self.model_name = _MODELS['biomedclip']['backbones'][self.vision_cls]
        self.model , _ = open_clip.create_model_from_pretrained(
                                        _MODELS['biomedclip']['backbones'][self.vision_cls],
                                        )
        self.model.to(device)
        self.model.eval()

    def text_features(self, input_text: List[str]) -> torch.Tensor:
        tokenizer = open_clip.get_tokenizer(self.model_name)
        inputs = [tokenizer(text).to(next(self.model.parameters()).device, non_blocking=True) for text in input_text]
        inputs = torch.cat(inputs)
        # preprocess text
        with torch.no_grad():
            text_features = self.model.encode_text(inputs)
        return text_features

class UniMedClipZeroShot(BaseZeroShotModel):
    def __init__(self, vision_cls: str = "a photo of a {}", device: str = "cuda", **kwargs):
        super().__init__(vision_cls, device)
        self.text_encoder_name = _MODELS['unimedclip']['text_encoder_name'][self.vision_cls]
        self.model = open_clip.create_model(
                                        _MODELS['unimedclip']['backbones'][self.vision_cls],
                                        _MODELS['unimedclip']['pretrained_weights'][self.vision_cls],
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

# taken from 
class BiomedCLIPViT_LoRA(nn.Module):
    MODEL_TAG = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

    def __init__(self, lora_rank=4):
        super().__init__()
        self.lora_rank = lora_rank
        biomedclip = open_clip.create_model(self.MODEL_TAG)
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_TAG)
        # LoRA-tune the vision transformer
        vit = biomedclip.visual
        # assert isinstance(vit, timm_ViT)
        self.lora_vit = LoRA_ViT_timm(vit_model=vit, r=lora_rank)
        self.text_encoder = biomedclip.text
    

    # get features from the vision transformer
    def forward(self, text, image):
        image_features = self.encode_image(image, normalize=True) 
        text_features = self.encode_text(text, normalize=True) 
        image_logits = image_features @ text_features.T 
        return image_logits
    
    def text_features(self, input_text: List[str]) -> torch.Tensor:
        inputs = [self.tokenizer(text).to(next(self.lora_vit.parameters()).device) for text in input_text]
        inputs = torch.cat(inputs, dim=0)
        # preprocess text
        with torch.no_grad():
            text_features = self.encode_text(inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, images: torch.Tensor, normalize=False) -> torch.Tensor:
        image_features = self.lora_vit(images)
        return F.normalize(image_features, dim=-1) if normalize else image_features
    
    def encode_text(self, text: torch.Tensor, normalize=False) -> torch.Tensor:
        text_features = self.text_encoder(text)
        return F.normalize(text_features, dim=-1) if normalize else text_features

class MedCLIPResnet_LoRA(nn.Module):
    def __init__(self, lora_rank=4):
        super().__init__()
        self.lora_rank = lora_rank
        medclip = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        # Load pretrained weights for MedCLIP
        medclip.from_pretrained(input_dir=_MODELS['medclip']['pretrained_weights']['resnet'])
        # LoRA-tune the vision transformer
        resnet = medclip.vision_model
        self.lora_resnet = LoRA_resnet(resnet, r=lora_rank)
        self.text_encoder = medclip.text_model
        self.tokenizer = AutoTokenizer.from_pretrained(_MODELS['medclip']['text_encoder_name']['resnet'])

    def forward(self, text, image):
        image_features = self.encode_image(image, normalize=True) 
        text_features = self.encode_text(text, normalize=True)
        # Ensure dimensions match before matrix multiplication
        if image_features.shape[-1] != text_features.shape[-1]:
            # Project to same dimension if needed
            projection = nn.Linear(image_features.shape[-1], text_features.shape[-1], bias=False).to(image_features.device)
            image_features = projection(image_features)
            
        image_logits = image_features @ text_features.T 
        return image_logits
    
    def text_features(self, input_text: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(next(self.lora_resnet.parameters()).device) for key, val in inputs.items()}
        # preprocess text
        with torch.no_grad():
            text_features = self.text_encoder(inputs["input_ids"], inputs["attention_mask"])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, images: torch.Tensor, normalize=False) -> torch.Tensor:
        image_features = self.lora_resnet(images)
        # Check if image_features is a dictionary (common in HF models)
        if isinstance(image_features, dict) and "pooler_output" in image_features:
            image_features = image_features["pooler_output"]
        return F.normalize(image_features, dim=-1) if normalize else image_features
    
    def encode_text(self, text_inputs, normalize=False) -> torch.Tensor:
        # Handle different input types
        if isinstance(text_inputs, dict):
            text_features = self.text_encoder(text_inputs["input_ids"], text_inputs["attention_mask"])
        else:
            text_features = self.text_encoder(text_inputs)
        
        # Handle different return types
        if isinstance(text_features, dict) and "pooler_output" in text_features:
            text_features = text_features["pooler_output"]
            
        return F.normalize(text_features, dim=-1) if normalize else text_features

class RobustMedClip(BaseZeroShotModel):
    def __init__(self, vision_cls: str = None, device: str = "cuda", **kwargs):
        super().__init__(vision_cls, device)

        self.lora_rank = kwargs.get('lora_rank', 4)
        
        if self.vision_cls == 'vit':
            self.model = BiomedCLIPViT_LoRA(lora_rank=self.lora_rank)
            self.tokenizer = open_clip.get_tokenizer(_MODELS['biomedclip']['backbones'][self.vision_cls])
        elif self.vision_cls == 'resnet':
            self.model = MedCLIPResnet_LoRA(lora_rank=self.lora_rank)
            
        else:
            raise ValueError(f"Unsupported backbone: {self.vision_cls}")

        # freeze the text encoder
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        
        pretrained_weights = _MODELS['rmedclip']['pretrained_weights'].get(self.vision_cls, None)
        if pretrained_weights and kwargs.get('load_pretrained', False):
            self.load(pretrained_weights)
        
        # self.model.to(device)
    
    def save(self, output_path):
        # save the model
        
        lora_weight_path = os.path.join(output_path, "lora_weights.safetensors")
        model_path = os.path.join(output_path, "model.pth")
        
        if self.vision_cls == 'vit':
            self.model.lora_vit.save_lora_parameters(lora_weight_path)
        elif self.vision_cls == 'resnet':
            self.model.lora_resnet.save_lora_parameters(lora_weight_path)
        else:
            raise ValueError(f"Unsupported backbone: {self.vision_cls}")
        
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        lora_weight_path = os.path.join(os.path.dirname(model_path), "lora_weights.safetensors")
        if self.vision_cls == 'vit':
            self.model.lora_vit.load_lora_parameters(lora_weight_path)
        elif self.vision_cls == 'resnet':
            self.model.lora_resnet.load_lora_parameters(lora_weight_path)
        else:
            raise ValueError(f"Unsupported backbone: {self.vision_cls}")
    
    def text_features(self, input_text: List[str]) -> torch.Tensor:
        return self.model.text_features(input_text)
    
    def batch_predict(self, images: torch.Tensor, text_features) -> np.ndarray:
        """Predict using the student model"""
        images = images.to(self.device)
        text_features = text_features.to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Ensure dimensions match before matrix multiplication
            if image_features.shape[-1] != text_features.shape[-1]:
                # Project to same dimension if needed
                projection = nn.Linear(image_features.shape[-1], text_features.shape[-1], bias=False).to(image_features.device)
                image_features = projection(image_features)
                
            logits_per_image = image_features @ text_features.T
            logits_per_image = torch.nn.functional.softmax(logits_per_image, dim=1).float()
        
        return logits_per_image
    

# example usage
if __name__ == "__main__":
    # Simple test case for RobustMedClip initialization
    print("Testing RobustMedClip initialization...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = RobustMedClip(device=device)

    test_prompts = ["a chest x-ray of pneumonia", "a chest x-ray of normal", "a chest x-ray of covid19"]
    text_features = model.text_features(test_prompts)
