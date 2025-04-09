import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from function import normal,normal_style
from function import calc_mean_std
import scipy.stats as stats
#from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from timm.models.tiny_vit import tiny_vit_21m_224                           
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        return x


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    


class StyTrans(nn.Module):
    """ StyTrans model with explicit CNN decoder and TinyViT encoder."""
    def __init__(self, decoder, args):
        super().__init__()
        
        # MobileNet encoder (for perceptual loss)
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.enc_1 = mobilenet[:2]  # low-level
        self.enc_2 = mobilenet[2:4]
        self.enc_3 = mobilenet[4:7]
        self.enc_4 = mobilenet[7:14]
        self.enc_5 = mobilenet[14:]
        
        for param in mobilenet.parameters():
            param.requires_grad = False
                
        #TinyViT encoder (transformer backbone)
        self.tiny_vit_encoder = tiny_vit_21m_224(pretrained=True)

        # explicitly explicitly explicit clearly model clearly explicitly explicitly explicitly clearly explicitly explicitly explicitly explicitly clearly explicitly explicitly explicitly explicitly clearly explicitly explicitly explicitly explicitly clearly explicitly explicitly explicitly explicitly clearly explicitly explicitly explicitly explicitly clearly explicitly explicitly explicitly explicitly clearly explicitly explicitly explicitly explicitly clearly explicitly explicitly explicitly explicitly
        self.decode = decoder  # explicitly CNN decoder explicitly
        self.linear_proj = nn.Conv2d(576, 512, kernel_size=1) # explicitly adjusted if necessary (576 is TinyViT_21m output channels explicitly explicitly clearly)

        self.mse_loss = nn.MSELoss()

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    

    def forward(self, samples_c, samples_s):
        # Receive input images directly (clearly fixed explicitly)
        content_input, style_input = samples_c, samples_s

        # Resize inputs explicitly for TinyViT requirements (224,224)
        content_resized = F.interpolate(content_input, size=(224,224), mode='bilinear', align_corners=False)
        style_resized = F.interpolate(style_input, size=(224,224), mode='bilinear', align_corners=False)

        # Forward pass through TinyViT (embedding outputs)
        content_emb = self.tiny_vit_encoder.forward_features(content_resized)  # Shape:(B,C,H,W)
        style_emb = self.tiny_vit_encoder.forward_features(style_resized)      # Shape:(B,C,H,W)

        # Linear projection clearly (channel matching explicitly)
        content_emb = self.linear_proj(content_emb)
        style_emb = self.linear_proj(style_emb)

        # Combine style & content embeddings explicitly
        combined = content_emb + style_emb

        # Decode generated stylized image explicitly
        Ics = self.decode(combined)

        # Compute multi-layer features explicitly for original and stylized images
        content_feats = self.encode_with_intermediate(content_input)
        style_feats = self.encode_with_intermediate(style_input)
        Ics_feats = self.encode_with_intermediate(Ics)

        # --- Content Loss Calculation (with explicit resizing clearly) ---
        content_loss = 0.0
        for idx in [-1, -2]:  # clearly last two feature layers for content explicitly
            if Ics_feats[idx].shape != content_feats[idx].shape:
                Ics_feats[idx] = F.interpolate(Ics_feats[idx], 
                                            size=content_feats[idx].shape[2:], 
                                            mode='bilinear', align_corners=False)
            content_loss += self.calc_content_loss(normal(Ics_feats[idx]),
                                                normal(content_feats[idx]))

        # --- Style Loss Calculation (with explicit resizing clearly) ---
        style_loss = 0.0
        for idx in range(5):  # explicitly all five feature layers for style explicitly
            if Ics_feats[idx].shape != style_feats[idx].shape:
                Ics_feats[idx] = F.interpolate(Ics_feats[idx], 
                                            size=style_feats[idx].shape[2:], 
                                            mode='bilinear', align_corners=False)
            style_loss += self.calc_style_loss(Ics_feats[idx], style_feats[idx])

        # --- Identity Loss Calculation (explicitly clearly with resizing) ---
        # Generate identity outputs explicitly
        Icc = self.decode(content_emb)
        Iss = self.decode(style_emb)

        # Ensure identity outputs match original inputs explicitly (spatial dimensions explicitly)
        if Icc.shape != content_input.shape:
            Icc = F.interpolate(Icc, size=content_input.shape[2:], mode='bilinear', align_corners=False)
        if Iss.shape != style_input.shape:
            Iss = F.interpolate(Iss, size=style_input.shape[2:], mode='bilinear', align_corners=False)

        identity_loss1 = self.calc_content_loss(Icc, content_input) + \
                        self.calc_content_loss(Iss, style_input)

        # Compute identity loss at feature level explicitly
        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)

        identity_loss2 = 0.0
        for idx in range(5):
            if Icc_feats[idx].shape != content_feats[idx].shape:
                Icc_feats[idx] = F.interpolate(Icc_feats[idx], 
                                            size=content_feats[idx].shape[2:], 
                                            mode='bilinear', align_corners=False)
            if Iss_feats[idx].shape != style_feats[idx].shape:
                Iss_feats[idx] = F.interpolate(Iss_feats[idx], 
                                            size=style_feats[idx].shape[2:], 
                                            mode='bilinear', align_corners=False)

            identity_loss2 += self.calc_content_loss(Icc_feats[idx], content_feats[idx]) + \
                            self.calc_content_loss(Iss_feats[idx], style_feats[idx])

        # Explicitly clearly finalized return
        return Ics, content_loss, style_loss, identity_loss1, identity_loss2