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


def get_timestep_embedding(
    timesteps: torch.Tensor,                # shape: [B, S]
    embedding_dim: int,                     # embedding dim per scalar
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert timesteps.ndim == 2, "Input must be of shape [B, S]"
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

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",  # or "squaredcos_cap_v2" for cosine beta schedule
            prediction_type="epsilon",  # the usual choice in DDPM training
        )

        # Freeze diffusion model
        for param in self.base.unet.parameters():
            param.requires_grad = False

    def forward(self, source_images, au_diffs, fake_images):
        prompt_embeds = self.au_processor(au_diffs)
        out = self.base(
            image=source_images,
            target=fake_images,
            ddpm_scheduler=self.scheduler,
            prompt_embeds=prompt_embeds,
            guidance_scale=1.0,
            num_inference_steps=1000,
        )
        # print(out)
        return out
    

class CustomStableDiffusionPipeline(StableDiffusionInstructPix2PixPipeline):
    # @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image = None,
        target = None,  # new
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        ddpm_scheduler: DDPMScheduler = None,  # new
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cross_attention_kwargs = None,
    ):
        
        callback_steps = None

        if ddpm_scheduler is not None:
            self.scheduler = ddpm_scheduler  # replace self.scheduler with ddpm scheduler
        
        # 0. Check inputs
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        device = self._execution_device

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. set timesteps
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

        # 5. Prepare Image latents
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
        )
        

        target_latents = self.prepare_image_latents(
            target,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
        )
        noise = torch.randn_like(target_latents)
        noisy_target_latents = self.scheduler.add_noise(target_latents, noise, timesteps)

        

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        num_channels_latents = noisy_target_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents + num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )


        # 8.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
       
        # concat latents, image_latents in the channel dimension
        scaled_latent_model_input = self.scheduler.scale_model_input(noisy_target_latents, timesteps)
        scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

        # predict the noise residual
        noise_pred = self.unet(
            scaled_latent_model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        self.maybe_free_model_hooks()
        return noise, noise_pred


# --- Training Loop ---
def train(args):
    import os
    os.environ["WANDB_DISABLE_SERVICE"] = "True"
    wandb.init(
        project="au-guided-diffusion",
        config=args,
        name=f"run-{wandb.util.generate_id()}",
        mode="offline"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==> loading dataset")
    dataset = AUImagePairDataset(args.dir)
    print(dataset[0]["source"].shape)
    print(dataset[1]["target"].shape)
    print(dataset[12]["au_diff"].shape)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("==> instantiating model")
    pipe = CustomStableDiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    au_module = AUToPromptEmbed().to(device)
    model = AUPix2PixPipeline(pipe, au_module)

    print("=== Trainable Parameters ===")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    print("==> setting up optimizer")
    optimizer = torch.optim.Adam(model.au_processor.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(args.epochs), desc="epoch"):
        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="batch", leave=False):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            au_diff = batch["au_diff"].to(device)

            noise_pred, noise = model(source, au_diff, target)

            loss = loss_fn(noise_pred.float(), noise.float()).float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        torch.save(au_module.state_dict(), f"./checkpoints/au_module_checkpoint_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune AU2PromptEmbed preprocesser from image pairs in a directory.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing .png images")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train(args)
