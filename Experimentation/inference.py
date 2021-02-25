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

import util.utils as util
from src.approaches.train_audio2landmark import Audio2landmark_model
from src.approaches.train_image_translation import Image_translation_block
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor