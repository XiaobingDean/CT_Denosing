import glob
import numpy as np
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.customTransform import *
import os

import medmnist
from medmnist import INFO, Evaluator

class customDatasetReader(Dataset):
    def __init__(self, dataset_name, mode, height, width, transformation=True):

        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        self.dataset = DataClass(split=mode, download=True)

        self.transformLR = transforms
        self.imageH = height
        self.imageW = width
        self.normalize = transforms.Normalize(normMean, normStd)
        self.var = 0.1
        self.mean = 0.0
        self.pov = 0.3


    def __len__(self):
        return (len(self.dataset))
    
    def __getitem__(self, i):


        self.transformHRGT = transforms.Compose([
                                                transforms.ToTensor(),
                                                self.normalize,
                                                ])

        self.gtImage = np.asarray(self.dataset[i][0]) / 255.

        # Transforms Images for training 
        self.gtImageHR = self.transformHRGT(self.gtImage)
        
        #Noise Modeling
        sigma = random.uniform(0, self.var ** self.pov)
        noiseModel = torch.clamp(torch.randn(self.gtImageHR.size()).uniform_(0, 1.) * sigma  + 0., 0., 1.)
    
        self.inputImage = self.gtImageHR + noiseModel
        
        return self.inputImage.float(), noiseModel.float()
