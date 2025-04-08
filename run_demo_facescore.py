
import os
import cv2 
import argparse
from PIL import Image
import torch
import torch.nn as nn
from networks.generator import Generator
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from FaceScore.FaceScore import FaceScore


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


def video2imgs(videoPath):
    cap = cv2.VideoCapture(videoPath)    
    judge = cap.isOpened()               
    img = []
    while judge:
        flag, frame = cap.read()         
        if not flag:
            break
        else:
           img.append(frame) 
    cap.release()

    return img

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

        self.face_score_model = FaceScore('checkpoints/FS_model.pt', med_config='checkpoints/med_config.json')

        print('==> loading data')
        self.save_path = args.output_folder
        os.makedirs(self.save_path, exist_ok=True)

        # load source video
        source_video = video2imgs(args.s_path)   
        # preprocess     
        self.source = []
        for i in source_video:
            img = Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))
            self.source.append(img_preprocessing(img,256).cuda())
            
    def run(self):
        # choose a random frame from source video as source img and expression
        self.source_img = random.choice(self.source)
        self.exp_img = random.choice(self.source)

        print('==> running')
        with torch.no_grad():
            exp_img = self.exp_img
            source_img = self.source_img

            # get expression latents, make sure they differ a lot
            exp_sim = self.gen.compare_expression_latents(source_img, exp_img)

            if exp_sim > 0.97:
                print(f"Ignored frame pairs with exp sim {exp_sim:.4f}")
                return None, None, None, np.nan, exp_sim
            
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
            face_score, _, __ = self.face_score_model.get_reward_from_img(fake)
            print(f'The face score is {face_score}')
        return source_img, exp_img, fake, face_score, exp_sim

    def run_batch(self):
        exp_sim_list = []
        fs_list = []
        for i in tqdm(range(self.args.n_samples)):
            source_img, exp_img, fake_img, fs, exp_sim = self.run()
            fs_list.append(fs)
            exp_sim_list.append(exp_sim)
            if np.isnan(fs): 
                continue

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            titles = ["Source Image", "Expression Image", "Generated Image"]
            images = [source_img, exp_img, fake_img]
            for ax, img, title in zip(axes, images, titles):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")

            plt.tight_layout()
            plt.subplots_adjust(top=0.90, bottom=0.10) 
            fig.text(0.5, 0.03, f"Expression Similarity: {exp_sim:.4f}, Generated FaceScore: {fs:.4f}", 
                    ha='center', fontsize=14, color='gray')
            plt.savefig(os.path.join(self.save_path, f"comparison_{i}.png"))
            plt.close()

        summary = {
            'Mean': [np.nanmean(exp_sim_list), np.nanmean(fs_list)],
            'Median': [np.nanmedian(exp_sim_list), np.nanmedian(fs_list)],
            'Std Dev': [np.nanstd(exp_sim_list), np.nanstd(fs_list)],
            'Max': [np.nanmax(exp_sim_list), np.nanmax(fs_list)],
            'Min': [np.nanmin(exp_sim_list), np.nanmin(fs_list)],
            'Count': [np.count_nonzero(~np.isnan(exp_sim_list)), np.count_nonzero(~np.isnan(fs_list))]
        }
        df = pd.DataFrame(summary, index=["Expression Similarities", "FaceScore"])
        print(df.round(4).to_string())


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
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    demo.run_batch()
