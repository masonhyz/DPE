import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import DDPMScheduler
# from diffusers import PipelineImageInput
# from diffusers.models.embeddings import get_timestep_embedding
from tqdm import tqdm
import argparse
import pandas as pd
import math
import torch.nn.functional as F
from typing import Union, List, Optional
import wandb
import matplotlib.pyplot as plt


def get_timestep_embedding(
    timesteps: torch.Tensor,                # shape: [B, S]
    embedding_dim: int,                     # embedding dim per scalar
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert timesteps.ndim == 2, f"Input must be of shape [B, S], found {timesteps.shape}"
    B, S = timesteps.shape
    half_dim = embedding_dim // 2

    # Compute the base exponent vector (same for all scalars)
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb_freqs = torch.exp(exponent)  # [half_dim]

    # Expand timesteps and compute sinusoid inputs
    # timesteps: [B, S] → [B, S, half_dim]
    args = timesteps.unsqueeze(-1).float() * emb_freqs  # [B, S, half_dim]
    emb = scale * args

    # Apply sin and cos → [B, S, half_dim * 2]
    sin_emb = torch.sin(emb)
    cos_emb = torch.cos(emb)
    emb = torch.cat([sin_emb, cos_emb], dim=-1)  # [B, S, embedding_dim]

    # Optionally flip sin and cos
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, :, half_dim:], emb[:, :, :half_dim]], dim=-1)

    # Pad if needed for odd embedding dim
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    # Flatten to [B, S * embedding_dim]
    return emb


class AUToPromptEmbed(nn.Module):
    def __init__(self, embed_dim=768, input_dim=12, freq_shift=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.freq_shift = freq_shift

        # Optional: Learnable projection after fixed sinusoidal
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, au_diff):
        # Get sinusoidal positional embedding
        # print(au_diff.shape)
        sinusoid = get_timestep_embedding(
            timesteps=au_diff,
            embedding_dim=self.embed_dim,
            downscale_freq_shift=self.freq_shift,
            flip_sin_to_cos=False,
            scale=1
        )
        # print(sinusoid.shape)
        # Apply trainable transformation
        return self.projection(sinusoid)


# --- Custom Dataset ---
class AUImagePairDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.aus = self._load_aus()
        self.pairs = self._find_pairs()
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        self.n_aus = 12  # There are 12 AU intensity fields

    def _load_aus(self):
        csv_path = os.path.join(self.image_dir, "extracted_aus.csv")
        df = pd.read_csv(csv_path)
        df.set_index("filename", inplace=True)
        return df

    def _is_nonzero_file(self, filepath):
        return os.path.isfile(filepath) and os.path.getsize(filepath) > 0

    def _find_pairs(self):
        files = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
        prefix_map = {}

        for f in files:
            if "_source.png" in f:
                key = f.replace("_source.png", "")
                prefix_map.setdefault(key, {})["source"] = f
            elif "_fake.png" in f:
                key = f.replace("_fake.png", "")
                prefix_map.setdefault(key, {})["fake"] = f

        pairs = []
        for key, pair in prefix_map.items():
            if "source" in pair and "fake" in pair:
                source_path = os.path.join(self.image_dir, pair["source"])
                fake_path = os.path.join(self.image_dir, pair["fake"])
                if self._is_nonzero_file(source_path) and self._is_nonzero_file(fake_path):
                    pairs.append((pair["source"], pair["fake"]))
                else:
                    print(f"[SKIP] Skipping pair with 0-byte file: {pair['source']} or {pair['fake']}")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_file, fake_file = self.pairs[idx]
        source_path = os.path.join(self.image_dir, source_file)
        fake_path = os.path.join(self.image_dir, fake_file)

        source_img = Image.open(source_path).convert("RGB")
        fake_img = Image.open(fake_path).convert("RGB")

        source_tensor = self.transform(source_img)
        fake_tensor = self.transform(fake_img)

        try:
            source_aus = self.aus.loc[source_file].values.astype(float)
            fake_aus = self.aus.loc[fake_file].values.astype(float)
            au_diff = torch.tensor(fake_aus - source_aus, dtype=torch.float32)
        except KeyError:
            print(f"[WARNING] AU data missing for {source_file} or {fake_file}, using random.")
            au_diff = torch.randn(self.n_aus)

        return {
            "source": source_tensor,
            "target": fake_tensor,
            "au_diff": au_diff
        }


# --- Full Training Pipeline ---
class AUPix2PixPipeline(nn.Module):
    def __init__(self, base_pipeline, au_processor):
        super().__init__()
        self.base = base_pipeline
        self.au_processor = au_processor

        # Freeze diffusion model
        for param in self.base.unet.parameters():
            param.requires_grad = False

    def forward(self, source_images, au_diffs):
        prompt_embeds = self.au_processor(au_diffs)
        out = self.base(
            image=source_images,
            prompt_embeds=prompt_embeds,
            guidance_scale=1.0,
            num_inference_steps=1000,
        )
        return out.images
    

def to_numpy(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.max() > 1:  # if image is in [0, 255], normalize to [0, 1]
            img = img / 255.0
        return img.permute(1, 2, 0).numpy()
    return img


# --- Training Loop ---
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==> loading dataset")
    dataset = AUImagePairDataset(args.dir)

    print("==> instantiating model")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    au_module = AUToPromptEmbed().to(device)
    au_module.load_state_dict(torch.load(args.au_checkpoint))
    model = AUPix2PixPipeline(pipe, au_module)

    model.eval()
    with torch.no_grad():
        print("=== Trainable Parameters ===")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        print("==> setting up optimizer")

        source = dataset[0]["source"].unsqueeze(0).to(device)  # no .half()
        au_diff = dataset[0]["au_diff"].unsqueeze(0).to(device)
        target = dataset[0]["target"]

        import torchvision.transforms.functional as TF
        source_img = TF.to_pil_image(source.squeeze(0).cpu())  # convert to PIL

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                generated = model(source_img, au_diff)

        print(generated.shape, target.shape)
        target_img = to_numpy(target)
        generated_img = to_numpy(generated)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(target_img)
        axs[0].set_title("Target")
        axs[0].axis("off")

        axs[1].imshow(generated_img)
        axs[1].set_title("Generated")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()
        plt.save_fig("comparison.png")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune AU2PromptEmbed preprocesser from image pairs in a directory.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing .png images")
    parser.add_argument("--au_checkpoint", type=str, required=True, help="Path to au preprocessing module checkpoint")

    args = parser.parse_args()
    train(args)
