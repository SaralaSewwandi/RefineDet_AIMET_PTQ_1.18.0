from aimet_torch.model_validator.model_validator import ModelValidator
import sys
import os
import argparse
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.networks.resnet_refinedet import ResNetRefineDet
from libs.utils.config import voc320, voc512, coco320, coco512, MEANS
from libs.data_layers.transform import detection_collate, BaseTransform
from libs.data_layers.roidb import combined_roidb, get_output_dir
from libs.data_layers.blob_dataset import BlobDataset

def validate_example_model():

    # Load the model to validate
        
    cfg = (voc320, voc512)[0]
    refinedet320 = VGGRefineDet(cfg['num_classes'], cfg)
    refinedet320.create_architecture()
    
    refinedet320.load_state_dict(torch.load('/home/ava/sarala/RefineDet_PreTrained_Checkpoints/vgg16_refinedet320_voc_120000.pth'))
    refinedet320.eval()
    refinedet320 = refinedet320.to(device='cuda')

    # Output of ModelValidator.validate_model will be True if model is valid, False otherwise
    ModelValidator.validate_model(refinedet320, model_input=torch.rand(32, 3,320, 320).cuda())
    

validate_example_model()