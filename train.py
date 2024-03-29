import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
from decoder.utils.utils import *
from model import Network
from config import params
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader_part import ViPCDataLoader
from dataloader import ViPCDataLoader2
from dataloader_final import ViPCDataLoader_ft, rotate_pc_on_cam_torch, rotate_back_pc_on_cam_torch
from dataloader_final import save_img, save_point_cloud
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from vis_utils import mix_shapes_2
from renderer import Renderer
from zero123_utils import Zero123
from tky_model import PointCloud

import torchvision
my_zero123 = Zero123(device=torch.device("cuda:0"))
tky_model = PointCloud(2048).to(torch.device("cuda:0"))
opt = params()

if opt.cat != None:

    CLASS = opt.cat
else:
    CLASS = 'plane'


MODEL = 'FIXED_RENDERING'
FLAG = 'train'
DEVICE = 'cuda:0'
VERSION = '0.0'
BATCH_SIZE = int(opt.batch_size)
MAX_EPOCH = int(opt.n_epochs)
EVAL_EPOCH = int(opt.eval_epoch)
RESUME = True


TIME_FLAG = time.asctime(time.localtime(time.time()))
CKPT_RECORD_FOLDER = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/record'
CKPT_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt.pth'
CONFIG_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/CONFIG.txt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gaussian_blur = torchvision.transforms.GaussianBlur(21, sigma=0.1)

# test the gaussian blur
# img = torchvision.io.read_image("image.png").unsqueeze(0).float()
# img = img / 255
# # img = torch.zeros(1, 3, 224, 224)
# # img[0, 0, 100, 100] = 1
# # img[0, 1, 100, 100] = 1
# # img[0, 2, 100, 100] = 1
# img = gaussian_blur(img)
# # img = 1 - img
# # print(img)
# save_img(img[0], "tkys_gaussian_blur.png")
# raise Exception("break")


if __name__ == "__main__":
    # store temp files
    if os.path.exists("__tmp__"):
        os.system("rm -rf __tmp__")
    os.makedirs("__tmp__")

def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
                continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def save_record(epoch, prec1, net: nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict,
               os.path.join(CKPT_RECORD_FOLDER, f'epoch{epoch}_{prec1:.4f}.pth'))


def save_ckpt(epoch, net, optimizer_all):
    ckpt = dict(
        epoch=epoch,
        model=net.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
    )
    torch.save(ckpt, CKPT_FILE)


def set_seed(seed=random.randint(1,10000)):
    if seed is not None:
        print(f"selected seed = {seed}")
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True


def weights_init_normal(m):
    """ Weights initialization with normal distribution.. Xavier """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_one_step(data, optimizer, network):

    image = data[0].to(device)
    partial = data[2].to(device)
    partpart = data[3].to(device)
    # for pretraining they're not using the GT PC, but also not using rendering guidance
    # first train the model using partpart method
    partial = farthest_point_sample(partial, 2048)    

    partpart = partpart.permute(0, 2, 1)
    partial = partial.permute(0, 2, 1)

    mixed, mixed_gt, mixed_img = mix_shapes_2(partpart, partial, image)

    mixed_gt = mixed_gt.permute(0, 2, 1)
    
    partial = partial.permute(0,2,1)
    batch_gt = torch.cat((mixed_gt, partial), dim = 0)
    batch_input = torch.cat((mixed, partpart), dim = 0)
    batch_view = torch.cat((mixed_img, image), dim = 0)

    complete = network(batch_input, batch_view)
    loss_total = loss_cd(batch_gt, complete)
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    return loss_total

def train_one_step_render(data, optimizer, network, renderer):
    global step
    global total_step

    step += 1
    # second train the model using partpart method + view rendering
    image = data[0].to(device)
    partial = data[2].to(device)
    partpart = data[5].to(device)
    eye = data[3].to(device)
    mask_gt = data[4].to(device)
    # view_rgb = data[6].to(device)
    view_metadata = data[7].to(device)
    
    partial = farthest_point_sample(partial, 2048)
    # save_point_cloud(partial[0], "__tmp__/partial.ply")
    partpart = partpart.permute(0, 2, 1)
    partial = partial.permute(0, 2, 1)
    # combine two shapes arbitrarily

    mixed, mixed_gt, mixed_img = mix_shapes_2(partpart, partial, image)

    # mixed_gt = mixed_gt.permute(0, 2, 1)
    
    partial = partial.permute(0,2,1)
    batch_gt = partial
    batch_input = partpart
    batch_view = image

    ### using the original network
    complete = network(batch_input, batch_view)
    complete, colors = complete[:, :, :3], complete[:, :, 3:6].clip(0, 1)
    
    ### using tky model
    # complete, colors = tky_model()

    # blur the image by 3x3 gaussian kernel
    # image = gaussian_blur(image)


    # render the completed shape
    # partial = partial
    proj = renderer(complete, eye, colors).clip(0, 1)
    proj = proj.permute(0,3,1,2)
    # 1. try white background by using 1 - image
    image = 1 - image
    proj = 1 - proj

    # 2. try gaussian blur
    image = gaussian_blur(image)
    proj = gaussian_blur(proj)

    save_img(proj[0], "__tmp__/proj.png")
    save_img(image[0], "__tmp__/image.png")
    save_point_cloud(complete[0], "__tmp__/complete.ply")
    save_point_cloud(partial[0], "__tmp__/partial.ply")
    # rotate completed by 90, 180, 270

    # check zero123 generated images
    
    # images = torch.cat([image] * 4, dim=0)
    # elevation = torch.tensor([40] * 4)
    # azimuth = torch.tensor([0, 90, 180, 270])
    # radius = torch.tensor([0] * 4)
    # my_zero123.get_img_embeds(images)
    # images = my_zero123.refine(images, elevation, azimuth, radius, strength=0, default_elevation=0)
    # images_cpu = images.cpu()
    # images_cpu = images_cpu * 255
    # images_cpu = images_cpu.type(torch.uint8)
    # for i in range(4):
    #     torchvision.io.write_png(images_cpu[i], f"__tmp__/tky_{i}.png")

    # rotate
    my_zero123.get_img_embeds(image)
    loss_rot = 0

    # sample a random elevation and azimuth
    
    elevation = torch.randint(-40, 40, (1,)).item()
    azimuth = torch.randint(0, 360, (1,)).item()

    # print(elevation, azimuth)

    my_ele = view_metadata[0][1]
    my_azi = view_metadata[0][0]
    original_pc = rotate_back_pc_on_cam_torch(complete, my_ele, my_azi)

    rotate_again = rotate_pc_on_cam_torch(original_pc, my_ele - elevation, my_azi - azimuth)
    rendered_rand_view = renderer(rotate_again, eye, colors).permute(0,3,1,2)
    rendered_rand_view = 1 - rendered_rand_view
    rendered_rand_view = gaussian_blur(rendered_rand_view)
    save_rendered_rand_view = rendered_rand_view[0].detach().cpu()

    # get target rand_view image
    # target_image = my_zero123.refine(image.unsqueeze(0), [elevation], [azimuth], [0], strength=0)
    save_img(save_rendered_rand_view, "__tmp__/save_rendered_rand_view.png")
    # save_img(target_image.squeeze(), "__tmp__/save_target_rand_view.png")

    # print(target_image.shape)
    # raise Exception("break")
    print(step / total_step)
    loss_rot += my_zero123.train_step(rendered_rand_view, [elevation] * image.shape[0], [azimuth] * image.shape[0], [0] * image.shape[0], step_ratio=step / total_step)
    # complete_180 = rotate_pc_on_cam_torch(complete, 20, 180)
    # save_point_cloud(complete_180[0], "__tmp__/complete_180.ply")
    # proj_180 = renderer(complete_180, eye, colors)
    # save_png_180 = proj_180.permute(0,3,1,2)[0].detach().cpu()
    # save_img(save_png_180, "__tmp__/proj_180.png")
    # loss_rot += my_zero123.train_step(proj_180.permute(0, 3, 1, 2), [0] * view_rgb.shape[0], [180] * view_rgb.shape[0], [0] * view_rgb.shape[0])
    # complete_270 = rotate_pc_on_cam_torch(complete, 40, 270)
    # save_point_cloud(complete_270[0], "__tmp__/complete_270.ply")
    # proj_270 = renderer(complete_270, eye, colors)
    # save_png_270 = proj_270.permute(0,3,1,2)[0].detach().cpu()
    # save_img(save_png_270, "__tmp__/proj_270.png")

    # # raise Exception("break")
    # loss_rot += my_zero123.train_step(proj_270.permute(0, 3, 1, 2), [40] * view_rgb.shape[0], [270] * view_rgb.shape[0], [0] * view_rgb.shape[0], step_ratio=0.002)
    
    # proj_partial = renderer(partial, eye)
    # proj_partial = proj_partial.permute(0,3,1,2).squeeze(1)

    # difference = np.abs(proj[0].detach().cpu().numpy() -  mask_gt[0].cpu().numpy())
    # plt.imsave('difference.png', difference)

    # plt.imsave('zao_img.png', image[0].permute(1,2,0).detach().cpu().numpy())
    # plt.imsave('zao.png', proj[0].detach().cpu().numpy(), cmap = plt.get_cmap("binary"))
    # plt.imsave('zao_partial.png', proj_partial[0].detach().cpu().numpy(), cmap = plt.get_cmap("binary"))
    # plt.imsave('zao_GT.png', mask_gt[0].cpu().numpy(), cmap= plt.get_cmap("binary"))
    H = np.array(np.mat('0.000009501, 0.000056320, 0.000215654, 0.000544067, 0.000929938, 0.001107336, 0.000929938, 0.000544067, 0.000215654, 0.000056320, 0.000009501; 0.000056320, 0.000313537, 0.001107336, 0.002543353, 0.003994979, 0.004589996, 0.003994979, 0.002543353, 0.001107336, 0.000313537, 0.000056320; 0.000215654, 0.001107336, 0.003454844, 0.006607893, 0.008327918, 0.008509345, 0.008327918, 0.006607893, 0.003454844, 0.001107336, 0.000215654; 0.000544067, 0.002543353, 0.006607893, 0.008265356, 0.002299816, -0.002872123, 0.002299816, 0.008265356, 0.006607893, 0.002543353, 0.000544067; 0.000929938, 0.003994979, 0.008327918, 0.002299816, -0.022397153, -0.039158923, -0.022397153, 0.002299816, 0.008327918, 0.003994979, 0.000929938; 0.001107336, 0.004589996, 0.008509345, -0.002872123, -0.039158923, -0.062876027, -0.039158923, -0.002872123, 0.008509345, 0.004589996, 0.001107336; 0.000929938, 0.003994979, 0.008327918, 0.002299816, -0.022397153, -0.039158923, -0.022397153, 0.002299816, 0.008327918, 0.003994979, 0.000929938; 0.000544067, 0.002543353, 0.006607893, 0.008265356, 0.002299816, -0.002872123, 0.002299816, 0.008265356, 0.006607893, 0.002543353, 0.000544067; 0.000215654, 0.001107336, 0.003454844, 0.006607893, 0.008327918, 0.008509345, 0.008327918, 0.006607893, 0.003454844, 0.001107336, 0.000215654; 0.000056320, 0.000313537, 0.001107336, 0.002543353, 0.003994979, 0.004589996, 0.003994979, 0.002543353, 0.001107336, 0.000313537, 0.000056320; 0.000009501, 0.000056320, 0.000215654, 0.000544067, 0.000929938, 0.001107336, 0.000929938, 0.000544067, 0.000215654, 0.000056320, 0.000009501'))
    H = torch.from_numpy(H).to(device).unsqueeze(0).unsqueeze(0).float() 

    # this is an edge detector
    edge = F.conv2d(mask_gt.unsqueeze(1), H, padding='same').squeeze(1)  
    tau = 0.02 
    edge_map = torch.where(edge>tau, 0.4, 1.0) 

    # proj = torch.max(proj, dim=1)[0]
    # edge_map = edge_map.unsqueeze(1).repeat(1, 3, 1, 1)
    loss_img = F.mse_loss(proj, image, reduction='mean')
    # loss_img = torch.mean(((proj-mask_gt)*edge_map)**2)     # loss_img is only calulated on the edge pixels
    loss_pc, _, _ = calc_dcd(complete, batch_gt)
    loss_pc= loss_pc.mean()

    loss_final = 0*loss_pc + 0*loss_img + 1e-3 * loss_rot
    print(loss_pc.item(), loss_img.item(), loss_rot.item())

    
    optimizer.zero_grad()
    loss_final.backward()
    optimizer.step()

    return loss_final



best_loss = 99999
best_epoch = 0
resume_epoch = 0
board_writer = SummaryWriter(
    comment=f'{MODEL}_{VERSION}_{BATCH_SIZE}_{FLAG}_{CLASS}_{TIME_FLAG}')

model = Network().apply(weights_init_normal)
colors = torch.ones(size=(opt.batch_size, 2048, 1)).to(device)
colors[:, :, 0] = colors[:, :, 0]*0.8
render = Renderer(colors).to(device)


loss_cd =  L2_ChamferLoss_weighted() 
loss_cd_eval = L2_ChamferEval()
loss_MSE = nn.MSELoss()
optimizer = torch.optim.Adam(filter(
    lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999))

ViPCDataset_train = ViPCDataLoader(
    'dataset/train_list2.txt', data_path=opt.dataroot, status="train", category=opt.cat)
train_loader = DataLoader(ViPCDataset_train,
                          batch_size=opt.batch_size,
                          num_workers=opt.nThreads,
                          shuffle=True,
                          drop_last=True)

# fine-tune dataset should be one class only
ViPCDataset_train_res = ViPCDataLoader_ft(
    'dataset/train_list_clean_plane.txt', data_path=opt.dataroot, status="train", category=opt.cat)
train_loader_res = DataLoader(ViPCDataset_train_res,
                          batch_size=opt.batch_size,
                          num_workers=opt.nThreads,
                          shuffle=True,
                          drop_last=True)

step = 0
total_step = len(ViPCDataset_train_res)


ViPCDataset_test = ViPCDataLoader2(
    'dataset/test_list2.txt', data_path=opt.dataroot, status="test", category=opt.cat)
test_loader = DataLoader(ViPCDataset_test,
                         batch_size=opt.batch_size,
                         num_workers=opt.nThreads,
                         shuffle=True,
                         drop_last=True)


if RESUME:
    ckpt_path = "ckpt_39.pt"
    ckpt_dict = torch.load(ckpt_path)
    model_state_dict = ckpt_dict['model_state_dict']
    # model_state_dict = {k: v for k, v in model_state_dict.items() if 'decoder' not in k or 'conv4' not in k}
    # model.load_state_dict(model_state_dict)
    load_my_state_dict(model, model_state_dict)
    
    optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
    resume_epoch = ckpt_dict['epoch']
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


if not os.path.exists(os.path.join(CKPT_RECORD_FOLDER)):
    os.makedirs(os.path.join(CKPT_RECORD_FOLDER))

# writing configuration
with open(CONFIG_FILE, 'w') as f:
    f.write('RESUME:'+str(RESUME)+'\n')
    f.write('FLAG:'+str(FLAG)+'\n')
    f.write('DEVICE:'+str(DEVICE)+'\n')
    f.write('BATCH_SIZE:'+str(BATCH_SIZE)+'\n')
    f.write('MAX_EPOCH:'+str(MAX_EPOCH)+'\n')
    f.write('CLASS:'+str(CLASS)+'\n')
    f.write('VERSION:'+str(VERSION)+'\n')
    f.write(str(opt.__dict__))


model.train()
model.to(device)

print('--------------------')
print('Training Starting')
print(f'Training Class: {CLASS}')
print('--------------------')

# set_seed(1)
opt.lr = 0.0001

for epoch in range(resume_epoch, resume_epoch + opt.n_epochs+1):
    Loss = 1e9
    # if epoch % EVAL_EPOCH == 0: 
        
    #     with torch.no_grad():
    #         model.eval()
    #         i = 0
    #         Loss = 0
    #         for data in tqdm(test_loader):

    #             i += 1
    #             image = data[0].to(device)
    #             partial = data[2].to(device)
    #             gt = data[1].to(device)
                

    #             partial = farthest_point_sample(partial, 2048)
    #             gt = farthest_point_sample(gt, 2048)

    #             partial = partial.permute(0, 2, 1)

    #             complete = model(partial, image)[:, :, :3]
    #             loss = loss_cd_eval(complete, gt)
                
    #             Loss += loss

    #         Loss = Loss/i
    #         board_writer.add_scalar(
    #             "Average_Loss_epochs_test", Loss.item(), epoch)

    #         if Loss < best_loss:
    #             best_loss = Loss
    #             best_epoch = epoch
    #         print(best_epoch, ' ', best_loss)

    #     print('****************************')
    #     print("test loss: \t", Loss, "\tbest: \t", best_loss, "\tbest epoch:\t", best_epoch)
    #     print('****************************')
    # if epoch % opt.ckp_epoch == 0: 

    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': Loss
    #     }, f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt_{epoch}.pt')

    # for each epoch, train with partpart first
    # then train with partpart + view rendering
    opt.status = "train"

    
    Loss = 0
    Loss_final = 0
    Loss_missing = 0
    i = 0
    model.train()

    # partpart model training
    # for data in tqdm(train_loader):
    #     if i > 200:
    #         break
    #     loss = train_one_step(data, optimizer, network=model)
    #     i += 1
    #     if i % opt.loss_print == 0:
    #         board_writer.add_scalar("Loss_iteration", loss.item(
    #         ), global_step=i + epoch * len(train_loader))

    #     Loss += loss


    # # partpart + rendering model training
    for data in tqdm(train_loader_res):
        # if i > 400:
        #     break
        loss = train_one_step_render(data, optimizer, network=model, renderer=render)
        i += 1
        if i % opt.loss_print == 0:
            board_writer.add_scalar("Loss_iteration", loss.item(
            ), global_step=i + epoch * len(train_loader))
            
        Loss += loss
    
    Loss = Loss/i
    
    print(f"epoch {epoch}: Loss = {Loss}")
    
    board_writer.add_scalar("Average_Loss_epochs_final", Loss.item(), epoch)





print('Train Finished!!')
