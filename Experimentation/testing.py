
import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import util.utils as util
from scipy.signal import savgol_filter

from src.approaches.train_audio2landmark import Audio2landmark_model
import sounddevice as sd
from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
import soundfile as sf


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
        
    
if __name__=='__main__':
    audio(r'audio\tmp1.wav',r'weights/ckpt/ckpt_autovc.pth')