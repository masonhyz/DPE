
import os
import cv2 
import argparse
from PIL import Image
from insightface_backbone_conv import iresnet100
import torch
import torch.nn as nn
from networks.generator import Generator
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from FaceScore.FaceScore import FaceScore
from sklearn.metrics.pairwise import cosine_similarity


def load_image1(filename, size):
    img = filename.convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0

def img_preprocessing(img_path, size):
    img = load_image1(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


# def video2imgs(videoPath):
#     cap = cv2.VideoCapture(videoPath)    
#     judge = cap.isOpened()               
#     img = []
#     while judge:
#         flag, frame = cap.read()         
#         if not flag:
#             break
#         else:
#            img.append(frame) 
#     cap.release()

#     return img


def load_video_safe(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) == 500:
                break
        cap.release()
        return frames
    
    except Exception as e:
        print(f"Failed to load video {video_path}: {e}")
        return None


def crop_and_preprocess(face_score_model, frame, output_size=256, scale=1.5):
    _, box, confidence = face_score_model.get_reward_from_img(frame)
    if (not confidence) or confidence[0] < 0.9:
        print("No face detected")
        return None

    x1, y1, x2, y2 = map(int, box[0])  # assuming box[0] is the correct bbox
    w, h = x2 - x1, y2 - y1
    size = int(max(w, h) * scale)  # upscale the size

    # Get center of original box
    cx = x1 + w // 2
    cy = y1 + h // 2

    # Compute new square bounds
    x1_new = max(cx - size // 2, 0)
    y1_new = max(cy - size // 2, 0)
    x2_new = x1_new + size
    y2_new = y1_new + size

    # Clamp to image boundaries
    h_frame, w_frame, _ = frame.shape
    x2_new = min(x2_new, w_frame)
    y2_new = min(y2_new, h_frame)
    x1_new = max(x2_new - size, 0)
    y1_new = max(y2_new - size, 0)

    face_crop = frame[y1_new:y2_new, x1_new:x2_new]
    face_crop = cv2.resize(face_crop, (output_size, output_size))

    face_crop = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    face_crop = img_preprocessing(face_crop, 256).cuda()
    return face_crop


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        model_path = args.model_path
        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, weights_only=False, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        # load facescore model
        self.face_score_model = FaceScore('./checkpoints/FS_model.pt', med_config='checkpoints/med_config.json')
        # load arcface model
        ckpt = '/checkpoints/insightface_glint360k.pth'
        self.arcface = iresnet100().eval()
        info = self.arcface.load_state_dict(torch.load(ckpt))
        print(info)

        print('==> loading data')
        self.save_path = args.output_folder
        os.makedirs(self.save_path, exist_ok=True)
        self.source = load_video_safe(args.s_path)
        
    def run(self):
        # choose and preprocess random frame pair
        self.source_img = crop_and_preprocess(self.face_score_model, random.choice(self.source))
        self.exp_img = crop_and_preprocess(self.face_score_model, random.choice(self.source))
        if self.source_img is None or self.exp_img is None:
            print("preprocessing failed")
            return {"source": None,
                    "driving": None, 
                    "fake": None, 
                    "face_score": np.nan, 
                    "cos_sim": np.nan, 
                    "exp_sim": np.nan,
                    "euclidean": np.nan
            }

        print('==> running')
        with torch.no_grad():
            exp_img = self.exp_img
            source_img = self.source_img

            # get expression latents, make sure they differ a lot
            exp_sim = self.gen.compare_expression_latents(source_img, exp_img)

            if exp_sim > self.args.exp_threshold:
                print(f"Ignored frame pairs with exp sim {exp_sim:.4f}")
                return {"source": None,
                        "driving": None, 
                        "fake": None, 
                        "exp_sim": exp_sim,
                        "face_score": np.nan, 
                        "cos_sim": np.nan, 
                        "euclidean": np.nan
                }
            
            # transfer expression
            output_dict = self.gen(source_img, exp_img, 'exp')
            fake = output_dict
            fake = fake.cpu().clamp(-1, 1)

            # write outputs
            fake = fake[:,:3,:,:].clone().cpu().float().detach().numpy()
            fake = (np.transpose(fake, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            fake = fake.astype(np.uint8)[0]

            source_img = source_img[:,:3,:,:].clone().cpu().float().detach().numpy()
            source_img = (np.transpose(source_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            source_img = source_img.astype(np.uint8)[0]

            exp_img = exp_img[:,:3,:,:].clone().cpu().float().detach().numpy()
            exp_img = (np.transpose(exp_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            exp_img = exp_img.astype(np.uint8)[0]

        print("==> evaluating")

        with torch.no_grad():
            face_score = np.nan
            cos_sim = np.nan
            if self.args.eval in ["facescore", "both"]:
                # face score evaluation
                face_score, _, __ = self.face_score_model.get_reward_from_img(fake)
                print(f'The face score is {face_score}')
            if self.args.eval in ["arcface", "both"]:
                # arcface evaluation
                # calculate ids
                id_fake = self.arcface(torch.tensor(fake).float().permute(2,0,1).unsqueeze(0))
                id_source = self.arcface(torch.tensor(source_img).float().permute(2,0,1).unsqueeze(0))

                # compute cosine similarities
                id_fake = np.transpose(id_fake.numpy()[0], (1,0))
                id_source = np.transpose(id_source.numpy()[0], (1,0))
                euclidean_dist = np.mean(np.linalg.norm(id_fake - id_source, axis=1))
                cos_sim_matrix = cosine_similarity(id_fake, id_source)
                cos_sim = np.mean(np.diag(cos_sim_matrix))

        return {"source": source_img,
                "driving": exp_img, 
                "fake": fake, 
                "face_score": face_score, 
                "euclidean": euclidean_dist,
                "cos_sim": cos_sim, 
                "exp_sim": exp_sim
        }

    def run_batch(self):
        qualified_path = os.path.join(self.save_path, "qualified")
        disqualified_path = os.path.join(self.save_path, "disqualified")
        os.makedirs(qualified_path, exist_ok=True)
        os.makedirs(disqualified_path, exist_ok=True)

        if not self.source:
            return
        euc_list = []
        exp_sim_list = []
        fs_list = []
        cos_sim_list = []

        base_name = os.path.splitext(os.path.basename(self.args.s_path))[0]
        for i in tqdm(range(self.args.n_samples), desc="Video"):
            try:
                res = self.run()
                fs_list.append(res["face_score"])
                exp_sim_list.append(res["exp_sim"])
                cos_sim_list.append(res["cos_sim"])
                euc_list.append(res["euclidean"])
                if res["source"] is None:
                    continue

                # comparison image
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                titles = ["Source Image", "Expression Image", "Generated Image"]
                images = [res["source"], res["driving"], res["fake"]]
                for ax, img, title in zip(axes, images, titles):
                    ax.imshow(img)
                    ax.set_title(title)
                    ax.axis("off")

                plt.tight_layout()
                plt.subplots_adjust(top=0.90, bottom=0.10)
                fig.text(0.5, 0.03, f"Expression Similarity: {res['exp_sim']:.4f}, Generated FaceScore: {res['face_score']:.4f}, Identity Similarity: {res['cos_sim']:.4f}, Euclidean: {res['euclidean']:.4f}", 
                        ha='center', fontsize=14, color='gray')
                plt.savefig(os.path.join(self.save_path, f"{base_name}_{i}_comp.png"))
                plt.close()
                
                # save data pair
                try:
                    if res["face_score"] > self.args.face_score_threshold and res["cos_sim"] > self.args.cos_sim_threshold and res["euclidean"] < self.args.euclidean_threshold:
                        save_folder = qualified_path
                    else:
                        save_folder = disqualified_path
                except:
                    save_folder = disqualified_path
                Image.fromarray(res["source"]).save(os.path.join(save_folder, f"{base_name}_{i}_source.png"))
                Image.fromarray(res["fake"]).save(os.path.join(save_folder, f"{base_name}_{i}_fake.png"))
            except:
                print(f"processing failed for {base_name}_{i}")
            
        # summary for batch
        summary = {
            'Mean': [np.nanmean(exp_sim_list), np.nanmean(fs_list), np.nanmean(cos_sim_list), np.nanmean(euc_list)],
            'Median': [np.nanmedian(exp_sim_list), np.nanmedian(fs_list), np.nanmedian(cos_sim_list), np.nanmedian(euc_list)],
            'Std Dev': [np.nanstd(exp_sim_list), np.nanstd(fs_list), np.nanstd(cos_sim_list), np.nanstd(euc_list)],
            'Max': [np.nanmax(exp_sim_list), np.nanmax(fs_list), np.nanmax(cos_sim_list), np.nanmax(euc_list)],
            'Min': [np.nanmin(exp_sim_list), np.nanmin(fs_list), np.nanmin(cos_sim_list), np.nanmin(euc_list)],
            'Count': [np.count_nonzero(~np.isnan(exp_sim_list)), np.count_nonzero(~np.isnan(fs_list)), np.count_nonzero(~np.isnan(cos_sim_list)), np.count_nonzero(~np.isnan(euc_list))]
        }
        df = pd.DataFrame(summary, index=["Expression Similarities", "FaceScore", "Identity Similarity (Cos)", "Identity Distance (Euclidean)"])
        print(df.round(4).to_string())
        df.to_csv(os.path.join(self.save_path, f"{base_name}_summary.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--s_path", type=str, default='./data/crop_video/4.mp4')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--output_folder", type=str, default='')
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--eval", type=str, default="both")
    parser.add_argument("--exp_threshold", type=float, default=0.97)
    parser.add_argument("--face_score_threshold", type=float, default=1.0)
    parser.add_argument("--cos_sim_threshold", type=float, default=0.95)
    parser.add_argument("--euclidean_threshold", type=float, default=20.0)
    args = parser.parse_args()

    if args.s_path.endswith('.mp4'):
        # run single video
        demo = Demo(args)
        demo.run_batch()
    else:
        # run full directory
        dir_path = args.s_path
        mp4_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
        print(dir_path)

        for mp4 in tqdm(mp4_files, desc="Directory"):
            print(dir_path)
            try:
                args.s_path = os.path.join(dir_path, mp4)
                print(f"==> Running on {args.s_path}")
                demo = Demo(args)
                demo.run_batch()
            except:
                print(f"{args.s_path} failed")