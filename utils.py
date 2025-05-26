# Based on LoRA-ViT: https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
# Modified by Haoran Wang.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter



_MODELS = {

    'clip': {
        'class_name': "ClipZeroShot",
        'backbones': {
                'vit': "ViT-B/16",
                'resnet': "RN50"
            },
        'pretrained_weights': {
                'vit': "",
                'resnet': ""
            },
        },
    'medclip': {
        'class_name': "MedclipZeroShot",
        'backbones': {
                'vit': "MedCLIPVisionModelViT",
                'resnet': "MedCLIPVisionModel"
            },
        'pretrained_weights': {
                'vit': "outputs/checkpoints/medclip/medclip-vit",
                'resnet': "outputs/checkpoints/medclip/medclip-resnet"
            },
        'text_encoder_name': {
                'vit': "emilyalsentzer/Bio_ClinicalBERT",
                'resnet': "emilyalsentzer/Bio_ClinicalBERT"
            },
        },
    'biomedclip': {
        'class_name': "BioMedClipZeroShot",
        'backbones': {
                'vit': "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            },
        'pretrained_weights': {
                'vit': "",
            },
        },
    'unimedclip': {
        'class_name': "UniMedClipZeroShot",
        'backbones': {
                'vit-B-16': "ViT-B-16-quickgelu",
                'vit-L-14': "ViT-L-14-336-quickgelu",
            },
        'pretrained_weights': {
                'vit-B-16': "./outputs/checkpoints/unimedclip/unimed_clip_vit_b16.pt",
                'vit-L-14': "./outputs/checkpoints/unimedclip/unimed_clip_vit_l14_large_text_encoder.pt"
            },
        'text_encoder_name': {
                'vit-B-16': "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                'vit-L-14': "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
            },
        },

    'rmedclip': {
        'class_name': "RobustMedClip",
        'backbones': {
                'vit': "",
                'resnet': "",
            },
        'pretrained_weights': {
                'vit': "/home/raza.imam/Documents/rmedclip/outputs/exp/vit/fewshot_10_percent/checkpoints/best_model/model.pth",
                'resnet': "/home/raza.imam/Documents/rmedclip/outputs/exp/resnet/fewshot_10_percent/checkpoints/best_model/model.pth"
            },
        }
}

class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_k = self.linear_b_k(self.linear_a_k(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, self.dim:-self.dim] += new_k
        qkv[:, :, -self.dim:] += new_v
        return qkv

class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, lora_layer=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.trunk.blocks)))

        # Create for storage, then we can init them or load weights
        # These are linear layers
        self.w_As = []
        self.w_Bs = []

        # Lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.trunk.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_k = nn.Linear(self.dim, r, bias=False)
            w_b_linear_k = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_k,
                w_b_linear_k,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.lora_vit = vit_model

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"LoRA ViT trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,})")
        print(f"Breakdown by component:")
        # Count parameters in each LoRA component
        w_as_params = sum(p.numel() for p in self.w_As if p.requires_grad)
        w_bs_params = sum(p.numel() for p in self.w_Bs if p.requires_grad)
        print(f"  - LoRA A matrices: {w_as_params:,} parameters")
        print(f"  - LoRA B matrices: {w_bs_params:,} parameters")
        print(f"  - Total LoRA parameters: {w_as_params + w_bs_params:,}")

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.proj.in_features
        _out = self.lora_vit.head.proj.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.proj.in_features
        _out = self.lora_vit.head.proj.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.proj.in_features
        _out = self.lora_vit.head.proj.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.proj.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.proj.in_features
            _out = self.lora_vit.head.proj.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.proj.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"LoRA ViT trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,})")

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)
    
class _LoRA_Conv2d(nn.Module):
    """LoRA implementation for Conv2d layers in ResNet"""
    def __init__(
        self,
        conv: nn.Conv2d,
        linear_a: nn.Module,
        linear_b: nn.Module,
    ):
        super().__init__()
        self.conv = conv
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dim = conv.in_channels
        self.out_channels = conv.out_channels
        # Store conv parameters to properly handle dimension changes
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.kernel_size = conv.kernel_size

    def forward(self, x):
        out = self.conv(x)
        # For LoRA in conv layers, we need to reshape the input
        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, self.dim)
        
        # Apply LoRA
        lora_out = self.linear_b(self.linear_a(x_reshaped))
        lora_out = lora_out.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)
        
        # Ensure dimensions match before adding
        if lora_out.shape != out.shape:
            # Adjust dimensions if needed (e.g., for strided convolutions)
            target_h, target_w = out.shape[2], out.shape[3]
            lora_out = F.interpolate(lora_out, size=(target_h, target_w), mode='nearest')
        
        # Add LoRA output to the original conv output
        return out + lora_out

class LoRA_resnet(nn.Module):
    def __init__(self, medclip_vision_model, r: int, lora_layer_patterns=None):
        super(LoRA_resnet, self).__init__()

        assert r > 0
        
        # Default: apply LoRA to all convolution layers in ResNet50
        if lora_layer_patterns is None:
            self.lora_layer_patterns = [
                "conv1",                      # First 7x7 conv layer
                "layer1\\..*\\.conv[123]",    # All convs in layer1 bottleneck blocks (1x1, 3x3, 1x1)
                "layer2\\..*\\.conv[123]",    # All convs in layer2 bottleneck blocks
                "layer3\\..*\\.conv[123]",    # All convs in layer3 bottleneck blocks
                "layer4\\..*\\.conv[123]",    # All convs in layer4 bottleneck blocks
            ]
        else:
            self.lora_layer_patterns = lora_layer_patterns

        # Create storage for LoRA parameters
        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()
        
        # Freeze the original model
        for param in medclip_vision_model.parameters():
            param.requires_grad = False
        
        # Apply LoRA to the model
        self._add_lora_layers(medclip_vision_model.model, r)
        
        self.lora_medclip = medclip_vision_model
        self.reset_parameters()

    def _add_lora_layers(self, model, r):
        import re
        
        # Function to check if a name matches any pattern
        def matches_pattern(name, patterns):
            for pattern in patterns:
                # Convert pattern to regex (already in regex format unlike previous glob patterns)
                if re.match(pattern, name):
                    return True
            return False
        
        # Recursive function to replace layers
        def replace_layers(module, full_name=''):
            for name, child in module.named_children():
                current_name = f"{full_name}.{name}" if full_name else name
                
                # Check if this is a Conv2d layer matching our patterns
                if isinstance(child, nn.Conv2d) and matches_pattern(current_name, self.lora_layer_patterns):
                    # Apply LoRA to all matching conv layers, not just 3x3
                    in_dim = child.in_channels
                    w_a = nn.Linear(in_dim, r, bias=False)
                    w_b = nn.Linear(r, child.out_channels, bias=False)
                    
                    self.w_As.append(w_a)
                    self.w_Bs.append(w_b)
                    
                    # Replace the conv with LoRA
                    setattr(module, name, _LoRA_Conv2d(child, w_a, w_b))
                else:
                    # Recursively process child modules
                    replace_layers(child, current_name)
        
        # Start the recursive process
        replace_layers(model)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        # Count parameters in the original model too for the total
        for _, param in self.lora_medclip.named_parameters():
            if not param.requires_grad:  # These are frozen parameters
                all_params += param.numel()
        
        print(f"LoRA ResNet trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,})")
        print(f"Breakdown by component:")
        
        # Count parameters in each LoRA component
        w_as_params = sum(p.numel() for p in self.w_As.parameters() if p.requires_grad)
        w_bs_params = sum(p.numel() for p in self.w_Bs.parameters() if p.requires_grad)
        
        print(f"  - LoRA A matrices: {w_as_params:,} parameters")
        print(f"  - LoRA B matrices: {w_bs_params:,} parameters")
        print(f"  - Total LoRA parameters: {w_as_params + w_bs_params:,}")

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_medclip.model.fc.in_features
        _out = self.lora_medclip.model.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_medclip.model.fc.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_medclip.model.fc.in_features
        _out = self.lora_medclip.model.fc.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_medclip.model.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """
        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_medclip.model.fc.in_features
        _out = self.lora_medclip.model.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_medclip.model.fc.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        load both lora and fc parameters.
        """
        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                try:
                    saved_tensor = f.get_tensor(saved_key)
                    w_A_linear.weight = Parameter(saved_tensor)
                except ValueError:
                    print(f"Could not load parameter {saved_key}")

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                try:
                    saved_tensor = f.get_tensor(saved_key)
                    w_B_linear.weight = Parameter(saved_tensor)
                except ValueError:
                    print(f"Could not load parameter {saved_key}")
                
            _in = self.lora_medclip.model.fc.in_features
            _out = self.lora_medclip.model.fc.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_medclip.model.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def forward(self, pixel_values, **kwargs):
        return self.lora_medclip(pixel_values, **kwargs)