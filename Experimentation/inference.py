import sys
from typing import final
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
from thirdparty.resemblyer_util.speaker_emb import get_spk_emb



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
        

def audio(audioPath,modelPath):
    au_data = []
    au_emb = []
    # Converting the audio to 16KHz to be on a safe side
    os.system('ffmpeg -y -loglevel error -i {} -ar 16000 audio/tmp.wav'.format(audioPath))
    shutil.copyfile('audio/tmp.wav', audioPath)

    # audio embedding
    me, _ = get_spk_emb(audioPath)
    au_emb.append(me.reshape(-1))

    print('Processing audio file')
    c = AutoVC_mel_Convertor('audio')

    au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=audioPath,autovc_model_path=modelPath)
    au_data += au_data_i
    if(os.path.isfile('audio/tmp.wav')):
        os.remove('audio/tmp.wav')
    return au_data

def landmarkPlacer(au_data):
    fl_data = []
    rot_tran, rot_quat, anchor_t_shape = [], [], []
    for au, info in au_data:
        au_length = au.shape[0]
        fl = np.zeros(shape=(au_length, 68 * 3))
        fl_data.append((fl, info))
        rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
        rot_quat.append(np.zeros(shape=(au_length, 4)))
        anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

    cleanStore('weights/dump/random_val_fl.pickle',fl_data)
    
    cleanStore('weights/dump/random_val_fl_interp.pickle',None)
    
    cleanStore('weights/dump/random_val_au.pickle',au_data)
    
    gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
    cleanStore('weights/dump/random_val_gaze.pickle',gaze)

        
def cleanStore(path,dumper):
    if os.path.exists(path):
        os.remove(path)
    try:
        if dumper==None:
            pass
        with open(path, 'wb') as fp:
            pickle.dump(dumper,fp)
    except Exception as e:
        print('The data is not written with the following error\n')
        print(e)
    finally:
        print(f'The file {os.path.basename(path)} is saved')
    
    return
        

    
if __name__ =='__main__':
    faceAlignment = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
    data = audio(r'audio\tmp1.wav',r'weights/ckpt/ckpt_autovc.pth')
    landmarkPlacer(data)
    
    
        
    