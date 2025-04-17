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
import random
from typing import Union, List, Optional


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
        print(au_diff.shape)
        sinusoid = get_timestep_embedding(
            timesteps=au_diff,
            embedding_dim=self.embed_dim,
            downscale_freq_shift=self.freq_shift,
            flip_sin_to_cos=False,
            scale=1
        )
        print(sinusoid.shape)
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
            output_type="pt"
        )
        # print(out)
        return out.images
    

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
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        ddpm_scheduler: DDPMScheduler = None,  # new
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cross_attention_kwargs = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept
                image latents as `image`, but if passing latents directly it is not encoded again.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                Push the generated image towards the initial `image`. Image guidance scale is enabled by setting
                `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
                linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
                value of at least `1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInstructPix2PixPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

        >>> image = download_image(img_url).resize((512, 512))

        >>> pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        ...     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "make the mountains snowy"
        >>> image = pipe(prompt=prompt, image=image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
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
        random_timestep = random.randint(0, num_inference_steps)  # random timestep
        self.scheduler.set_timesteps(random_timestep, device=device)  # put random timestep in scheduler
        timesteps = self.scheduler.timesteps

        # # 4. set timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps



        # 5. Prepare Image latents
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
        )

        # height, width = image_latents.shape[-2:]
        # height = height * self.vae_scale_factor
        # width = width * self.vae_scale_factor

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

        # # 6. Prepare latent variables
        # num_channels_latents = self.vae.config.latent_channels
        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )

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

        # # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

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
            timesteps[-1],
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # # perform guidance
        # if self.do_classifier_free_guidance:
        #     noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        #     noise_pred = (
        #         noise_pred_uncond
        #         + self.guidance_scale * (noise_pred_text - noise_pred_image)
        #         + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        #     )

        # # compute the previous noisy sample x_t -> x_t-1
        # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # if callback_on_step_end is not None:
        #     callback_kwargs = {}
        #     for k in callback_on_step_end_tensor_inputs:
        #         callback_kwargs[k] = locals()[k]
        #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

        #     latents = callback_outputs.pop("latents", latents)
        #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
        #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
        #     image_latents = callback_outputs.pop("image_latents", image_latents)

        


        
        # if not output_type == "latent":
        #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        #     image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        # else:
        #     image = latents
        #     has_nsfw_concept = None

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

        return noise, noise_pred



# --- Training Loop ---
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==> loading dataset")
    dataset = AUImagePairDataset(args.dir)
    print(dataset[0]["source"].shape)
    print(dataset[1]["target"].shape)
    print(dataset[12]["au_diff"].shape)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("==> instantiating model")
    pipe = CustomStableDiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    au_module = AUToPromptEmbed().to(device)
    model = AUPix2PixPipeline(pipe, au_module)

    print("=== Trainable Parameters ===")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    print("==> setting up optimizer")
    optimizer = torch.optim.Adam(model.au_processor.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(10), desc="epoch"):
        for batch in tqdm(dataloader, desc="batch"):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            au_diff = batch["au_diff"].to(device)
            print(source.shape, target.shape, au_diff.shape)

            noise_pred, noise = model(source, au_diff, target)
            # print(generated)
            # print("Generated requires grad:", generated.requires_grad)
            # print("Grad fn:", generated.grad_fn)
            loss = loss_fn(noise_pred.float(), noise.float()).float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune AU2PromptEmbed preprocesser from image pairs in a directory.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing .png images")
    args = parser.parse_args()
    train(args)
