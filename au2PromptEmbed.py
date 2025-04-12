import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.models.embeddings import get_timestep_embedding
import sys
from tqdm import tqdm
import argparse


# Add LibreFace to path
current_dir = os.path.dirname(os.path.abspath(__file__))
library_path = os.path.join(current_dir, "LibreFace")
sys.path.append(library_path)
from libreface import get_facial_attributes

# --- AU Preprocessing Module ---
class AUToPromptEmbed(nn.Module):
    def __init__(self, embed_dim=2048, input_dim=12, freq_shift=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.freq_shift = freq_shift

    def forward(self, au_diff):
        # au_diff: [B, input_dim]
        # Use sinusoidal timestep-style encoding
        sinusoid = get_timestep_embedding(
            timesteps=au_diff,  # shape [B, input_dim]
            embedding_dim=self.embed_dim,
            downscale_freq_shift=self.freq_shift,
            flip_sin_to_cos=False,
            scale=1
        )
        return sinusoid

# --- Custom Dataset ---
class AUImagePairDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.pairs = self._find_pairs()
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def _find_pairs(self):
        files = os.listdir(self.image_dir)
        pairs = []
        keys = set(f.split("_")[0] + "_" + f.split("_")[1] for f in files)
        for key in keys:
            source = f"{key}_source.jpeg"
            fake = f"{key}_fake.jpeg"
            if source in files and fake in files:
                pairs.append((source, fake))
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

        source_aus = get_facial_attributes(source_path).get("detected_aus", {})
        fake_aus = get_facial_attributes(fake_path).get("detected_aus", {})

        # If either set of AUs is missing, return random vector
        if not source_aus or not fake_aus:
            au_diff = torch.randn(12)
        else:
            keys = sorted(set(source_aus) & set(fake_aus))
            au_diff = torch.tensor([fake_aus[k] - source_aus[k] for k in keys], dtype=torch.float32)
            if len(au_diff) < 12:
                padding = torch.zeros(12 - len(au_diff))
                au_diff = torch.cat([au_diff, padding], dim=0)

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
            num_inference_steps=50
        )
        return out.images

# --- Training Loop ---
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==> loading dataset")
    dataset = AUImagePairDataset(args.dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print("==> instantiating model")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16).to(device)
    au_module = AUToPromptEmbed().to(device)
    model = AUPix2PixPipeline(pipe, au_module)

    print("==> setting up optimizer")
    optimizer = torch.optim.Adam(model.au_processor.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(10), desc="epoch"):
        for batch in tqdm(dataloader, desc="batch"):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            au_diff = batch["au_diff"].to(device)

            generated = model(source, au_diff)
            loss = loss_fn(generated, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune AU2PromptEmbed preprocesser from image pairs in a directory.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing .png images")
    args = parser.parse_args()
    train(args)
