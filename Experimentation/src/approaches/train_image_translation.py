"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

from src.models.model_image_translation import ResUnetGenerator, VGGLoss
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
import numpy as np
import cv2
import os, glob
from src.dataset.image_translation.image_translation_dataset import vis_landmark_on_img, vis_landmark_on_img98, vis_landmark_on_img74


from thirdparty.AdaptiveWingLoss.core import models
from thirdparty.AdaptiveWingLoss.utils.utils import get_preds_fromhm

import face_alignment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Image_translation_block():

    def __init__(self, add_audio_in,load_G_name, single_test=False):
        print('Run on device {}'.format(device))

        # for key in vars(opt_parser).keys():
        #     print(key, ':', vars(opt_parser)[key])
        self.add_audio_in = add_audio_in
        self.load_G_name = load_G_name

        # model
        if(self.add_audio_in):
            self.G = ResUnetGenerator(input_nc=7, output_nc=3, num_downs=6, use_dropout=False)
        else:
            self.G = ResUnetGenerator(input_nc=6, output_nc=3, num_downs=6, use_dropout=False)

        if (self.load_G_name != ''):
            ckpt = torch.load(self.load_G_name)
            try:
                self.G.load_state_dict(ckpt['G'])
            except:
                tmp = nn.DataParallel(self.G)
                tmp.load_state_dict(ckpt['G'])
                self.G.load_state_dict(tmp.module.state_dict())
                del tmp

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs in G mode!")
            self.G = nn.DataParallel(self.G)

        self.G.to(device)

    def single_test(self, jpg, fls, filename=None, prefix='', grey_only=False):
        import time
        st = time.time()
        self.G.eval()

        writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mjpg'), 62.5, (256 * 3, 256))

        for i, frame in enumerate(fls):

            img_fl = np.ones(shape=(256, 256, 3)) * 255
            fl = frame.astype(int)
            img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))
            frame = np.concatenate((img_fl, jpg), axis=2).astype(np.float32)/255.0

            image_in, image_out = frame.transpose((2, 0, 1)), np.zeros(shape=(3, 256, 256))
            # image_in, image_out = frame.transpose((2, 1, 0)), np.zeros(shape=(3, 256, 256))
            image_in, image_out = torch.tensor(image_in, requires_grad=False), \
                                  torch.tensor(image_out, requires_grad=False)

            image_in, image_out = image_in.reshape(-1, 6, 256, 256), image_out.reshape(-1, 3, 256, 256)
            image_in, image_out = image_in.to(device), image_out.to(device)

            g_out = self.G(image_in)
            g_out = torch.tanh(g_out)

            g_out = g_out.cpu().detach().numpy().transpose((0, 2, 3, 1))
            g_out[g_out < 0] = 0
            ref_in = image_in[:, 3:6, :, :].cpu().detach().numpy().transpose((0, 2, 3, 1))
            fls_in = image_in[:, 0:3, :, :].cpu().detach().numpy().transpose((0, 2, 3, 1))

            if(grey_only):
                g_out_grey =np.mean(g_out, axis=3, keepdims=True)
                g_out[:, :, :, 0:1] = g_out[:, :, :, 1:2] = g_out[:, :, :, 2:3] = g_out_grey


            for i in range(g_out.shape[0]):
                frame = np.concatenate((ref_in[i], g_out[i], fls_in[i]), axis=1)* 255.0 #g_out[i] 
                writer.write(frame.astype(np.uint8))

        writer.release()
        print('Time - only video:', time.time() - st)
        print(filename)
        print(filename[9:-16])
        print(filename[:-4])
        print(prefix)
        if(filename is None):
            filename = 'v'
        os.system('ffmpeg -loglevel error -y -i out.mp4 -i {} -pix_fmt yuv420p -strict -2 output/{}_{}.mp4'.format(
            'audio/'+filename[9:-16]+'.wav',
            prefix, filename[:-4]))
        # os.system('rm out.mp4')

        print('Time - ffmpeg add audio:', time.time() - st)





