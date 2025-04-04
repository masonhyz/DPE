
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


        exp_img = video2imgs(args.s_path)

        img = Image.open(args.s_path)


        exp = []
        for i in exp_img:
            img = Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))
            exp.append(img_preprocessing(img,256).cuda())

        self.pose_img = random.choice(exp)
        self.exp_img = random.choice(exp)

        self.run()


    def run(self):
        output_dir = self.save_path

        crop_vi = os.path.join(output_dir, 'edit.mp4')
        out_edit = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        crop_vi = os.path.join(output_dir, 's.mp4')
        out_s = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        crop_vi = os.path.join(output_dir, 'exp.mp4')
        out_exp = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        print('==> running')
        with torch.no_grad():
            img_exp = self.exp_img
            img_source = self.pose_img
            
            # transfer expression
            output_dict = self.gen(img_source, img_exp, 'exp')
            fake = output_dict
            fake = fake.cpu().clamp(-1, 1)

            # write outputs
            fake = fake[:,:3,:,:].clone().cpu().float().detach().numpy()
            fake = (np.transpose(fake, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            fake = fake.astype(np.uint8)[0]
            fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)
            out_edit.write(fake)

            img_source = img_source[:,:3,:,:].clone().cpu().float().detach().numpy()
            img_source = (np.transpose(img_source, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            img_source = img_source.astype(np.uint8)[0]
            img_source = cv2.cvtColor(img_source, cv2.COLOR_RGB2BGR)
            out_s.write(img_source)

            img_exp = img_exp[:,:3,:,:].clone().cpu().float().detach().numpy()
            img_exp = (np.transpose(img_exp, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            img_exp = img_exp.astype(np.uint8)[0]
            img_exp = cv2.cvtColor(img_exp, cv2.COLOR_RGB2BGR)
            out_exp.write(img_exp)

            out_edit.release()
            out_s.release()
            out_exp.release()


        with torch.no_grad():
            ckpt = 'checkpoints/insightface_glint360k.pth'
            arcface = iresnet100().eval()
            info = arcface.load_state_dict(torch.load(ckpt))
            print(info)

            fake = torch.tensor(fake).float().permute(2,0,1)
            fake = fake.unsqueeze(0)
            img_source = torch.tensor(img_source).float().permute(2,0,1).unsqueeze(0)
            id_fake = arcface(fake, return_id512=True)
            id_source = arcface(img_source, return_id512=True)

            print(id_fake.shape)
            print(id_source.shape)
            
            id_fake = np.transpose(id_fake.numpy()[0], (1,0))
            id_source = np.transpose(id_source.numpy()[0], (1,0))
            
            sims = cosine_similarity(id_fake, id_source)

            print(sims.shape)





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
