import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
import cv2
import pickle
import shutil
from scipy.signal import savgol_filter

import torch
import face_alignment

import imgResizing
import util.utils as util
from src.approaches.train_audio2landmark import Audio2landmark_model
from src.approaches.train_image_translation import Image_translation_block
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor



def imgSelection(imgPath,faceAlignment):
    image = cv2.imread(imgPath)
    if image.shape[0] != image.shape[1] or image.shape[0]!=256:
        image = imgResizing.imageResizing(image)
        print('[INFO]: The dimensions of the image is being altered to cater the model')
    
    shapes = faceAlignment.get_landmarks(image)[0]
    if (not shapes or len(shapes) != 1):
        print('Cannot detect face landmarks. Exit.')
        exit(-1)

    ''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
    shapes[49:54, 1] += 1.
    shapes[55:60, 1] -= 1.
    shapes[[37,38,43,44], 1] -=2
    shapes[[40,41,46,47], 1] +=2
        
if __name__ =='__main__':
    faceAlignment = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
    
    
        
    