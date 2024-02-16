from diffusers import DDIMScheduler
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kiui
import torchvision
import sys
import os
from tqdm import tqdm
from PIL import Image
sys.path.append('./')
def save_image(image, path):
    if torch.is_tensor(image):
        if image.shape[0] == 4:
            image[:3] = 1 - image[:3]
            image =  image.mul(255).byte().cpu().detach().numpy().transpose(1, 2, 0)

            image = Image.fromarray(image, 'RGBA')
            image.save(path, )
            return
    raise Exception("Not supported image type")

from zero123 import Zero123Pipeline
from dataloader_final import rotate_pc_on_cam

class Zero123(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/zero123-xl-diffusers"):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        assert self.fp16, 'Only zero123 fp16 is supported for now.'

        # model_key = "ashawkey/zero123-xl-diffusers"
        # model_key = './model_cache/stable_zero123_diffusers'

        # model
        self.pipe = Zero123Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        # stable-zero123 has a different camera embedding
        self.use_stable_zero123 = 'stable' in model_key

        self.pipe.image_encoder.eval()  # CLIP image encoder for context image conditioning
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.clip_camera_projection.eval() # a linear projection layer for CLIP embedding and camera embedding

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.embeddings = None

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(images=x_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=self.dtype)
        c = self.pipe.image_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.vae.config.scaling_factor
        # self.embeddings are the CLIP and VAE embeddings
        self.embeddings = [c, v]
    
    def get_cam_embeddings(self, elevation, azimuth, radius, default_elevation=0):
        if self.use_stable_zero123:
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), np.deg2rad([90 + default_elevation] * len(elevation))], axis=-1)
        else:
            # original zero123 camera embedding
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), radius], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(dtype=self.dtype, device=self.device) # [8, 1, 4]
        return T

    @torch.no_grad()
    def refine(self, pred_rgb, elevation, azimuth, radius, 
               guidance_scale=5, steps=50, strength=0.8, default_elevation=0,
        ):

        batch_size = pred_rgb.shape[0]

        self.scheduler.set_timesteps(steps)

        if strength == 0:
            init_step = 0
            latents = torch.randn((batch_size, 4, 32, 32), device=self.device, dtype=self.dtype)
        else:
            init_step = int(steps * strength)
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
        cc_emb = torch.cat([self.embeddings[0].unsqueeze(1), T], dim=-1)
        cc_emb = self.pipe.clip_camera_projection(cc_emb)
        cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)
        
        vae_emb = self.embeddings[1]
        vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            
            x_in = torch.cat([latents] * 2)
            t_in = t.view(1).to(self.device)

            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 256, 256]
        return imgs
    
    def train_step(self, pred_rgb, elevation, azimuth, radius, step_ratio=None, guidance_scale=5, as_latent=False, default_elevation=0):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        batch_size = pred_rgb.shape[0]

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)

            T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
            # cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
            cc_emb = torch.cat([self.embeddings[0].unsqueeze(1), T], dim=-1)
            cc_emb = self.pipe.clip_camera_projection(cc_emb)
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

            # vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
            vae_emb = self.embeddings[1]
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()

        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        # print(loss.shape)
        # for l in loss:
        #     for i in l[0]:
        #         print(i.item(), end=' ')
        #     print()
        # loss = loss.sum(dim=(1, 2, 3))

        return loss
    

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]
        # get the VAE latents from images

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents
    
    
def zero123_process(model123, image_path, angles=[[0, 0], [0, 90], [0, 180], [0, 270]]):
    '''
    For each image, create 3 images with different angles by using zero123
    '''
    device = torch.device('cuda')

    name, ext = os.path.splitext(image_path)

    # Load image black background
    image = torchvision.io.read_image(image_path)[:, ...].to(device).unsqueeze(0).contiguous().float() / 255
    
    # Load image white background
    image = kiui.read_image(image_path, mode='tensor')
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
    # print(image.shape)
    # save_image(image[0], "tky_original.png")

    # raise Exception("brak")
    # image = image.repeat(3, 1, 1, 1)
    model123 = model123.to(device)
    model123.get_img_embeds(image)

    for ele, azi in angles:
        outputs = model123.refine(image, elevation=[ele], azimuth=[azi], radius=[0.1], strength=0)
        outputs = F.interpolate(outputs, (137, 137), mode='bilinear', align_corners=False)
        outputs = outputs.cpu()
        outputs = outputs * 255
        outputs = outputs.type(torch.uint8)
        # create mask for the output
        mask = torch.sum(outputs, dim=1, keepdim=True) <= 254 * 3
        
        
        # mask = mask.type(torch.uint8) * 255
        # mask = mask.squeeze(0)
        # torchvision.io.write_png(mask, "tky_mask.png")
        torchvision.io.write_png(outputs[0], name + "_e%d_a%d.png" % (ele, azi))
        # raise Exception("break")

        # add the mask as the alpha channel
        # outputs = torch.cat([outputs, mask], dim=1)
        # save_image(outputs[0], name + "_e%d_a%d.png" % (ele, azi))

if __name__ == '__main__':
    # device = torch.device('cuda')
    # zero123 = Zero123(device, model_key='ashawkey/zero123-xl-diffusers')
    # train_list = open('dataset/train_list_clean_plane.txt', 'r').readlines()
    # # for ran_key in [train_list[0]]:
    # for ran_key in tqdm(train_list[2000:]):
    #     view_path = os.path.join("/media/tky135/data/ShapeNetViPC-Dataset/ShapeNetViPC-View/",ran_key.split(';')[0]+'/'+ran_key.split(';')[1]+'/rendering/'+ran_key.split(';')[-1].replace('\n','')+'.png')
    #     assert os.path.exists(view_path), view_path
    #     zero123_process(zero123, view_path)
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import kiui

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='__tmp__/image.png')
    parser.add_argument('--elevation', type=float, default=0, help='delta elevation angle in [-90, 90]')
    parser.add_argument('--azimuth', type=float, default=0, help='delta azimuth angle in [-180, 180]')
    parser.add_argument('--radius', type=float, default=0, help='delta camera radius multiplier in [-0.5, 0.5]')
    parser.add_argument('--stable', action='store_true')

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = kiui.read_image(opt.input, mode='tensor')
    if (len(image.shape) == 2):
        image = image.unsqueeze(2).repeat(1, 1, 3)
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)

    # image2 = kiui.read_image("18.png", mode='tensor')
    # if (len(image2.shape) == 2):
    #     image2 = image2.unsqueeze(2).repeat(1, 1, 3)
    # image2 = image2.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    # image2 = F.interpolate(image2, (256, 256), mode='bilinear', align_corners=False)
    # print(f'[INFO] loading model ...')
    
    if opt.stable:
        zero123 = Zero123(device, model_key='ashawkey/stable-zero123-diffusers')
    else:
        zero123 = Zero123(device, model_key='ashawkey/zero123-xl-diffusers')

    print(f'[INFO] running model ...')

    azimuth = opt.azimuth
    elevation = opt.elevation
    ### testing train_step
    images = torch.cat([image] * 4 + [image] * 4, dim=0)
    # change black background to white background
    image_mask = torch.sum(images, dim=1, keepdim=True) == 0
    images[image_mask.repeat(1, 3, 1, 1)] = 1
    # images = images * 255
    # images = images.type(torch.uint8).cpu()
    # for i in range(8):
    #     torchvision.io.write_png(images[i], f"__tmp__/tky_{i}.png")
    # raise Exception("break")
    zero123.get_img_embeds(images)
    elevation = torch.tensor([0] * 8)
    azimuth = torch.tensor([0, 90, 180, 270, 0, 90, 180, 270])
    radius = torch.tensor([0.1] * 8)
    zero123.get_img_embeds(images)
    images = zero123.refine(images, elevation, azimuth, radius, strength=0, default_elevation=0)
    images_cpu = images.cpu()
    images_cpu = images_cpu * 255
    images_cpu = images_cpu.type(torch.uint8)
    for i in range(8):
        torchvision.io.write_png(images_cpu[i], f"__tmp__/tky_{i}.png")

    # raise Exception("break")

    zero123.get_img_embeds(images[0].unsqueeze(0))
    images = images[[1]]
    elevation = [0]
    azimuth = [180]
    radius = [0]

    loss = zero123.train_step(images, elevation, azimuth, radius, step_ratio=0.002, default_elevation=0)
    print(loss)
    # print(loss)

    # image = images[[7]].to(device)
    # image.requires_grad = True
    # for i in range(100):
    #     loss = zero123.train_step(image , [0], [0], [0.1], step_ratio=None, default_elevation=0)
    #     loss.backward()
    #     image.data = image.data - 0.1 * image.grad
    #     image.grad.zero_()
    #     print(loss)
    # image = image.cpu()
    # image = image * 255
    # image = image.type(torch.uint8)
    # torchvision.io.write_png(image[0], f"__tmp__/tky_train.png")
    # loss = loss.cpu()
    # loss = loss * 255
    # loss = loss.type(torch.uint8)
    # # images = images.cpu()
    # # images = images * 255
    # # images = images.type(torch.uint8)
    # for i in range(8):
    #     torchvision.io.write_png(loss[i], f"__tmp__/tky_{i}.png")
    # while True:
    #     outputs = zero123.refine(image, elevation=[elevation], azimuth=[azimuth], radius=[opt.radius], strength=0)
    #     plt.imshow(outputs.float().cpu().numpy().transpose(0, 2, 3, 1)[0])
    #     plt.show()
    #     azimuth = (azimuth + 90) % 360
    #     # elevation = (elevation + 45) % 180 - 90
