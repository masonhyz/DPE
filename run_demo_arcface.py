
import os
import cv2 
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from networks.generator import Generator
import numpy as np
import torchvision
import random
from insightface_backbone_conv import iresnet100
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0

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


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)


def video2imgs(videoPath):
    cap = cv2.VideoCapture(videoPath)    
    judge = cap.isOpened()                
    fps = cap.get(cv2.CAP_PROP_FPS)     

    frames = 1                           
    count = 1                           
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

        # output_dir = self.save_path

        # crop_vi = os.path.join(output_dir, 'edit.mp4')
        # out_fake = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        # crop_vi = os.path.join(output_dir, 'source.mp4')
        # out_source = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        # crop_vi = os.path.join(output_dir, 'exp.mp4')
        # out_exp = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        print('==> running')
        with torch.no_grad():
            exp_img = self.exp_img
            source_img = self.source_img
            
            # transfer expression
            output_dict = self.gen(source_img, exp_img, 'exp')
            fake = output_dict
            fake = fake.cpu().clamp(-1, 1)

            # write outputs
            fake = fake[:,:3,:,:].clone().cpu().float().detach().numpy()
            fake = (np.transpose(fake, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            fake = fake.astype(np.uint8)[0]
            fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)
            # out_fake.write(fake)

            source_img = source_img[:,:3,:,:].clone().cpu().float().detach().numpy()
            source_img = (np.transpose(source_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            source_img = source_img.astype(np.uint8)[0]
            source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
            # out_source.write(source_img)

            exp_img = exp_img[:,:3,:,:].clone().cpu().float().detach().numpy()
            exp_img = (np.transpose(exp_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            exp_img = exp_img.astype(np.uint8)[0]
            exp_img = cv2.cvtColor(exp_img, cv2.COLOR_RGB2BGR)
            # out_exp.write(exp_img)

            # out_fake.release()
            # out_source.release()
            # out_exp.release()

        print("==> evaluating")
        with torch.no_grad():
            ckpt = 'checkpoints/insightface_glint360k.pth'
            arcface = iresnet100().eval()
            info = arcface.load_state_dict(torch.load(ckpt))
            print(info)

            # calculate ids
            id_fake = arcface(torch.tensor(fake).float().permute(2,0,1).unsqueeze(0))
            id_source = arcface(torch.tensor(source_img).float().permute(2,0,1).unsqueeze(0))

            # compute cosine similarities
            id_fake = np.transpose(id_fake.numpy()[0], (1,0))
            id_source = np.transpose(id_source.numpy()[0], (1,0))
            cos_sim_matrix = cosine_similarity(id_fake, id_source)
            cos_sim_scalar = np.mean(np.diag(cos_sim_matrix))
            print(cos_sim_scalar)
        
        return source_img, exp_img, fake, cos_sim_scalar

    def run_batch(self, n):
        for i in tqdm(range(n)):
            source_img, exp_img, fake_img, cos_sim_scalar = self.run()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f"Cosine Similarity: {cos_sim_scalar:.4f}", fontsize=16)

            titles = ["Source Image", "Expression Image", "Generated Image"]
            images = [source_img, exp_img, fake_img]

            for ax, img, title in zip(axes, images, titles):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")

            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # leave space for suptitle
            plt.savefig(os.path.join(self.save_path, f"comparison_{i}.png"))
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--s_path", type=str, default='./data/crop_video/4.mp4')
    parser.add_argument("--face", type=str, default='both')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--output_folder", type=str, default='')
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    demo.run_batch(10)
