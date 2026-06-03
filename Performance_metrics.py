"""Image-quality and semantic-similarity metrics for TokCom evaluation."""

import warnings

import clip
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch_msssim import ms_ssim

warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13")


def tensor_to_pil(img_tensor):
    """Convert a CHW image tensor in [0, 1] to a PIL image."""
    return T.ToPILImage()(img_tensor)


def calculate_psnr(img1, img2):
    """Calculate PSNR for image tensors whose pixel values are in [0, 1]."""
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr


def calculate_ms_ssim(img1, img2):
    """Calculate MS-SSIM for image tensors whose pixel values are in [0, 1]."""
    return ms_ssim(img1, img2, data_range=1.0, size_average=True)


def calculate_lpips(img1, img2, loss_fn_alex):
    """Calculate LPIPS distance for image tensors."""
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    with torch.no_grad():
        lpips_value = loss_fn_alex(img1, img2)

    if lpips_value.dim() > 0:
        return lpips_value.mean().item()
    return lpips_value.item()


def calculate_clip_score(img1, img2, model, preprocess):
    """Calculate average CLIP image-feature cosine similarity."""
    if img1.dim() == 4:
        clip_scores = []
        for i in range(img1.size(0)):
            image1 = preprocess(tensor_to_pil(img1[i])).unsqueeze(0).to(img1.device)
            image2 = preprocess(tensor_to_pil(img2[i])).unsqueeze(0).to(img1.device)

            with torch.no_grad():
                image1_features = model.encode_image(image1)
                image2_features = model.encode_image(image2)
                clip_scores.append(F.cosine_similarity(image1_features, image2_features).item())

        return np.mean(clip_scores)

    image1 = preprocess(tensor_to_pil(img1)).unsqueeze(0).to(img1.device)
    image2 = preprocess(tensor_to_pil(img2)).unsqueeze(0).to(img1.device)

    with torch.no_grad():
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)
        return F.cosine_similarity(image1_features, image2_features).item()
