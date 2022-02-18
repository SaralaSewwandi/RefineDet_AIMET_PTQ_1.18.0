

import torch
import torchvision

from libs.networks.vgg_refinedet import VGGRefineDet
from libs.utils.config import voc320, voc512

from torchvision import models
from aimet_torch.cross_layer_equalization import equalize_model



def cross_layer_equalization_auto():
    cfg = (voc320, voc512)[0]
    refinedet320 = VGGRefineDet(cfg['num_classes'], cfg)
    refinedet320.create_architecture()
    
    refinedet320.load_state_dict(torch.load('/home/ava/sarala/RefineDet_PreTrained_Checkpoints/vgg16_refinedet320_voc_120000.pth'))
    
    input_shape = (32, 3,320, 320)
    
    refinedet320.eval()

    # Performs BatchNorm fold, Cross layer scaling and High bias folding
    equalize_model(refinedet320, input_shape)
    
    
cross_layer_equalization_auto()