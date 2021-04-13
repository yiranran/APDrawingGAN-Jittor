import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2
import time

from models import *
from datasets import *
from utils import *

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="apdrawing", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--load_pre_train', type=int, default=1, help='whether load pre-trained model')
parser.add_argument('--load_pre_train_name', type=str, default="pre-training", help='the path to load pre-trained model')
parser.add_argument('--load_auxiliary_name', type=str, default="auxiliary", help='the path to load auxiliary model')
parser.add_argument('--nepoch', type=int, default=300, help='# of epoch at starting learning rate')
parser.add_argument('--nepoch_decay', type=int, default=0, help='# of epoch to linearly decay learning rate to zero')
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--in_channels", type=int, default=3, help="number of input channels")
parser.add_argument("--out_channels", type=int, default=1, help="number of output channels")
parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
parser.add_argument('--lambda_local', type=float, default=25.0, help='weight for Local loss')
parser.add_argument('--lambda_chamfer', type=float, default=0.1, help='weight for chamfer loss')
parser.add_argument('--lambda_chamfer2', type=float, default=0.1, help='weight for chamfer loss2')
parser.add_argument(
    "--sample_interval", type=int, default=400, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
parser.add_argument("--val_input_folder", type=str, default="./samples/A/example", help="the folder of input photos")
parser.add_argument("--val_lm_folder", type=str, default="./samples/A_landmark/example", help="the folder of input landmarks")
parser.add_argument("--val_mask_folder", type=str, default="./samples/A_mask/example", help="the folder of foreground landmarks")
opt = parser.parse_args()
print(opt)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)

def save_single_image(img, path):
    N,C,W,H = img.shape
    img = img[0]
    min_ = -1
    max_ = 1
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    cv2.imwrite(path,img)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("checkpoints/%s" % opt.dataset_name, exist_ok=True)

# Loss functions
criterion_GAN = nn.BCELoss() # no lsgan
criterion_pixelwise = nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
G_global = GeneratorUNet(in_channels=opt.in_channels, out_channels=opt.out_channels)
G_l_eyel = PartUnet(in_channels=opt.in_channels, out_channels=opt.out_channels)
G_l_eyer = PartUnet(in_channels=opt.in_channels, out_channels=opt.out_channels)
G_l_nose = PartUnet(in_channels=opt.in_channels, out_channels=opt.out_channels)
G_l_mouth = PartUnet(in_channels=opt.in_channels, out_channels=opt.out_channels)
G_l_hair = PartUnet2(in_channels=opt.in_channels, out_channels=opt.out_channels)
G_l_bg = PartUnet2(in_channels=opt.in_channels, out_channels=opt.out_channels)
G_combine = Combiner(in_channels=2*opt.out_channels, out_channels=opt.out_channels)
G_nets = [G_global, G_l_eyel, G_l_eyer, G_l_nose, G_l_mouth, G_l_hair, G_l_bg, G_combine]
D_global = Discriminator()
D_l_eyel = Discriminator()
D_l_eyer = Discriminator()
D_l_nose = Discriminator()
D_l_mouth = Discriminator()
D_l_hair = Discriminator()
D_l_bg = Discriminator()
D_nets = [D_global, D_l_eyel, D_l_eyer, D_l_nose, D_l_mouth, D_l_hair, D_l_bg]

if opt.load_pre_train != 0:
    # Load pretrained models using npr data
    gen_model_path = os.path.join("checkpoints", opt.load_pre_train_name, "latest_net_gen.pth")
    gen_state_dict = jt.safeunpickle(gen_model_path)
    G_global.load_state_dict(gen_state_dict['G'])
    G_l_eyel.load_state_dict(gen_state_dict['GLEyel'])
    G_l_eyer.load_state_dict(gen_state_dict['GLEyer'])
    G_l_nose.load_state_dict(gen_state_dict['GLNose'])
    G_l_mouth.load_state_dict(gen_state_dict['GLMouth'])
    G_l_hair.load_state_dict(gen_state_dict['GLHair'])
    G_l_bg.load_state_dict(gen_state_dict['GLBG'])
    G_combine.load_state_dict(gen_state_dict['GCombine'])
    dis_model_path = os.path.join("checkpoints", opt.load_pre_train_name, "latest_net_dis.pth")
    dis_state_dict = jt.safeunpickle(dis_model_path)
    D_global.load_state_dict(dis_state_dict['D'])
    D_l_eyel.load_state_dict(dis_state_dict['DLEyel'])
    D_l_eyer.load_state_dict(dis_state_dict['DLEyer'])
    D_l_nose.load_state_dict(dis_state_dict['DLNose'])
    D_l_mouth.load_state_dict(dis_state_dict['DLMouth'])
    D_l_hair.load_state_dict(dis_state_dict['DLHair'])
    D_l_bg.load_state_dict(dis_state_dict['DLBG'])

DT1 = GeneratorUNet(in_channels=1, out_channels=1, num_downs=9)
DT2 = GeneratorUNet(in_channels=1, out_channels=1, num_downs=9)
Line1 = GeneratorUNet(in_channels=1, out_channels=1, num_downs=9)
Line2 = GeneratorUNet(in_channels=1, out_channels=1, num_downs=9)
DT1.load(os.path.join("checkpoints", opt.load_auxiliary_name, "latest_net_DT1.pth"))
DT2.load(os.path.join("checkpoints", opt.load_auxiliary_name, "latest_net_DT2.pth"))
Line1.load(os.path.join("checkpoints", opt.load_auxiliary_name, "latest_net_Line1.pth"))
Line2.load(os.path.join("checkpoints", opt.load_auxiliary_name, "latest_net_Line2.pth"))

# Optimizers
G_nets_params = G_nets[0].parameters()
for net in G_nets[1:]:
    G_nets_params += net.parameters()
optimizer_G = jt.optim.Adam(G_nets_params, lr=opt.lr, betas=(opt.b1, opt.b2))
D_nets_params = D_nets[0].parameters()
for net in D_nets[1:]:
    D_nets_params += net.parameters()
optimizer_D = jt.optim.Adam(D_nets_params, lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
dataloader = ImageDataset("data/%s" % opt.dataset_name, load_h=opt.img_height, load_w=opt.img_width).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = TestDataset(opt.val_input_folder, opt.val_lm_folder, opt.val_mask_folder, mode="val", load_h=opt.img_height, load_w=opt.img_width).set_attrs(
    batch_size=10,
    shuffle=True,
    num_workers=1,
)

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    batches = next(iter(val_dataloader))

    real_A = batches[0]
    real_A_eyel = batches[1]
    real_A_eyer = batches[2]
    real_A_nose = batches[3]
    real_A_mouth = batches[4]
    real_A_hair = batches[5]
    real_A_bg = batches[6]
    mask = batches[7]
    mask2 = batches[8]
    center = batches[9]

    maskh = mask*mask2
    maskb = inverse_mask(mask2)

    fake_B0 = G_global(real_A)
    # EYES, NOSE, MOUTH
    fake_B_eyel = G_l_eyel(real_A_eyel)
    fake_B_eyer = G_l_eyer(real_A_eyer)
    fake_B_nose = G_l_nose(real_A_nose)
    fake_B_mouth = G_l_mouth(real_A_mouth)
    # HAIR, BG AND PARTCOMBINE
    fake_B_hair = G_l_hair(real_A_hair)
    fake_B_bg = G_l_bg(real_A_bg)
    fake_B1 = partCombiner2_bg(center, fake_B_eyel, fake_B_eyer, fake_B_nose, fake_B_mouth, fake_B_hair, fake_B_bg, maskh, maskb, comb_op=1, load_h=opt.img_height, load_w=opt.img_width)
    # FUSION NET
    fake_B = G_combine(jt.contrib.concat((fake_B0, fake_B1), 1))

    img_sample = np.concatenate([real_A.data, fake_B.repeat(1,3,1,1).data], -2)
    save_image(img_sample, "images/%s/%s.jpg" % (opt.dataset_name, batches_done), nrow=5)

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batches in enumerate(dataloader):

        real_A = batches[0]
        real_A_eyel = batches[1]
        real_A_eyer = batches[2]
        real_A_nose = batches[3]
        real_A_mouth = batches[4]
        real_A_hair = batches[5]
        real_A_bg = batches[6]
        mask = batches[7]
        mask2 = batches[8]
        center = batches[9]
        real_B = batches[10]
        dt1gt = batches[11]
        dt2gt = batches[12]

        maskh = mask*mask2
        maskb = inverse_mask(mask2)

        # Adversarial ground truths
        valid = jt.ones([real_A.shape[0], 1]).stop_grad()
        fake = jt.zeros([real_A.shape[0], 1]).stop_grad()

        fake_B0 = G_global(real_A)
        # EYES, NOSE, MOUTH
        fake_B_eyel = G_l_eyel(real_A_eyel)
        fake_B_eyer = G_l_eyer(real_A_eyer)
        fake_B_nose = G_l_nose(real_A_nose)
        fake_B_mouth = G_l_mouth(real_A_mouth)
        # HAIR, BG AND PARTCOMBINE
        fake_B_hair = G_l_hair(real_A_hair)
        fake_B_bg = G_l_bg(real_A_bg)
        fake_B1 = partCombiner2_bg(center, fake_B_eyel, fake_B_eyer, fake_B_nose, fake_B_mouth, fake_B_hair, fake_B_bg, maskh, maskb, comb_op=1, load_h=opt.img_height, load_w=opt.img_width)
        # FUSION NET
        fake_B = G_combine(jt.contrib.concat((fake_B0, fake_B1), 1))

        # ------------------
        #  Train Generators
        # ------------------
        # GAN loss
        pred_fake = D_global(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        fake_B_locals = getLocalParts(fake_B, center, maskh, maskb, load_h=opt.img_height, load_w=opt.img_width)
        real_A_locals = [real_A_eyel, real_A_eyer, real_A_nose, real_A_mouth, real_A_hair, real_A_bg]
        loss_GAN_local = 0
        for j in range(6):
            pred_fake_local = D_nets[j+1](fake_B_locals[j], real_A_locals[j])
            loss_GAN_local += criterion_GAN(pred_fake_local, valid)
        
        # L1 loss
        loss_pixel = criterion_pixelwise(fake_B, real_B) * opt.lambda_L1
        
        # DT loss
        ## 1) d_CM(a_i,G(p_i))
        fake_B_gray = fake_B
        real_B_gray = real_B

        dt1 = DT1(fake_B_gray)
        dt2 = DT2(fake_B_gray)
        dt1 = dt1/2.0+0.5#[-1,1]->[0,1]
        dt2 = dt2/2.0+0.5
        bs = real_B_gray.shape[0]
        real_B_gray_line1 = Line1(real_B_gray)
        real_B_gray_line2 = Line2(real_B_gray)
        loss_G_chamfer = (dt1[(real_B_gray<0)&(real_B_gray_line1<0)].sum() + dt2[(real_B_gray>=0)&(real_B_gray_line2>=0)].sum()) / bs * opt.lambda_chamfer

        ## 2) d_CM(G(p_i),a_i)
        fake_B_gray_line1 = Line1(fake_B_gray)
        fake_B_gray_line2 = Line2(fake_B_gray)
        loss_G_chamfer2 = (dt1gt[(fake_B_gray<0)&(fake_B_gray_line1<0)].sum() + dt2gt[(fake_B_gray>=0)&(fake_B_gray_line2>=0)].sum()) / bs * opt.lambda_chamfer2
        
        # Local loss
        real_B_locals = getLocalParts(real_B, center, maskh, maskb, load_h=opt.img_height, load_w=opt.img_width)
        loss_G_local = 0
        for j in range(6):
            loss_G_local += criterion_pixelwise(fake_B_locals[j], real_B_locals[j]) * opt.lambda_local
        
        # Total loss
        loss_G = loss_GAN + 0.25 * loss_GAN_local + loss_pixel + (loss_G_chamfer + loss_G_chamfer2) + loss_G_local
        optimizer_G.step(loss_G)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Real loss
        pred_real = D_global(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)
        loss_real_local = 0
        for j in range(6):
            pred_real_local = D_nets[j+1](real_B_locals[j], real_A_locals[j])
            loss_real_local += criterion_GAN(pred_real_local, valid)
        # Fake loss
        pred_fake = D_global(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)
        loss_fake_local = 0
        for j in range(6):
            pred_fake_local = D_nets[j+1](fake_B_locals[j].detach(), real_A_locals[j])
            loss_fake_local += criterion_GAN(pred_fake_local, fake)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake + loss_real_local + loss_fake_local)
        optimizer_D.step(loss_D)

        if warmup_times==-1:
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            jt.sync_all()
            if batches_done % 5 == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, dt: %f, local: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.numpy()[0],
                        loss_G.numpy()[0],
                        loss_GAN.numpy()[0],
                        loss_pixel.numpy()[0],
                        (loss_G_chamfer+loss_G_chamfer2).numpy()[0],
                        loss_G_local.numpy()[0],
                        time_left,
                    )
                )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                img_sample = np.concatenate([real_A.data, fake_B.repeat(1,3,1,1).data, real_B.repeat(1,3,1,1).data, fake_B0.repeat(1,3,1,1).data, fake_B1.repeat(1,3,1,1).data, (dt1*2-1).repeat(1,3,1,1).data, (dt2*2-1).repeat(1,3,1,1).data, (dt1gt*2-1).repeat(1,3,1,1).data, (dt2gt*2-1).repeat(1,3,1,1).data], -2)
                save_image(img_sample, "images/%s/train_%s.jpg" % (opt.dataset_name, batches_done), nrow=1)
                sample_images(batches_done)
        else:
            jt.sync_all()
            cnt += 1
            print(cnt)
            if cnt == warmup_times:
                jt.sync_all(True)
                sta = time.time()
            if cnt > warmup_times + run_times:
                jt.sync_all(True)
                total_time = time.time() - sta
                print(f"run {run_times} iters cost {total_time} seconds, and avg {total_time / run_times} one iter.")
                exit(0)
            
            if batches_done % opt.sample_interval == 0:
                img_sample = np.concatenate([real_A.data, fake_B.repeat(1,3,1,1).data, real_B.repeat(1,3,1,1).data, fake_B0.repeat(1,3,1,1).data, fake_B1.repeat(1,3,1,1).data, (dt1*2-1).repeat(1,3,1,1).data, (dt2*2-1).repeat(1,3,1,1).data, (dt1gt*2-1).repeat(1,3,1,1).data, (dt2gt*2-1).repeat(1,3,1,1).data], -2)
                save_image(img_sample, "images/%s/train_%s.jpg" % (opt.dataset_name, batches_done), nrow=1)
                sample_images(batches_done)
    if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
        # Save model checkpoints
        G_global.save("checkpoints/%s/%d_net_G_global.pkl" % (opt.dataset_name, epoch+1))
        G_l_eyel.save("checkpoints/%s/%d_net_G_l_eyel.pkl" % (opt.dataset_name, epoch+1))
        G_l_eyer.save("checkpoints/%s/%d_net_G_l_eyer.pkl" % (opt.dataset_name, epoch+1))
        G_l_nose.save("checkpoints/%s/%d_net_G_l_nose.pkl" % (opt.dataset_name, epoch+1))
        G_l_mouth.save("checkpoints/%s/%d_net_G_l_mouth.pkl" % (opt.dataset_name, epoch+1))
        G_l_hair.save("checkpoints/%s/%d_net_G_l_hair.pkl" % (opt.dataset_name, epoch+1))
        G_l_bg.save("checkpoints/%s/%d_net_G_l_bg.pkl" % (opt.dataset_name, epoch+1))
        G_combine.save("checkpoints/%s/%d_net_G_combine.pkl" % (opt.dataset_name, epoch+1))
        D_global.save("checkpoints/%s/%d_net_D_global.pkl" % (opt.dataset_name, epoch+1))
        D_l_eyel.save("checkpoints/%s/%d_net_D_l_eyel.pkl" % (opt.dataset_name, epoch+1))
        D_l_eyer.save("checkpoints/%s/%d_net_D_l_eyer.pkl" % (opt.dataset_name, epoch+1))
        D_l_nose.save("checkpoints/%s/%d_net_D_l_nose.pkl" % (opt.dataset_name, epoch+1))
        D_l_mouth.save("checkpoints/%s/%d_net_D_l_mouth.pkl" % (opt.dataset_name, epoch+1))
        D_l_hair.save("checkpoints/%s/%d_net_D_l_hair.pkl" % (opt.dataset_name, epoch+1))
        D_l_bg.save("checkpoints/%s/%d_net_D_l_bg.pkl" % (opt.dataset_name, epoch+1))
