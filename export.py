from __future__ import print_function
import torch
# Quantization related import
from aimet_torch.quantsim import QuantizationSimModel

import logging
from aimet_torch.cross_layer_equalization import equalize_model

from torch.utils.data import DataLoader
from torchvision import models
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_torch.utils import create_fake_data_loader
from aimet_torch.model_preparer import prepare_model




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


import pdb

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
        
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(
    description='RefineDet Test With Pytorch')
parser.add_argument('--dataset', default='voc', choices=['voc', 'coco'],
                    type=str, help='voc or coco')
parser.add_argument('--network', default='vgg16',
                    help='Base network')
parser.add_argument('--input_size', default=320, type=int,
                    help='Input size for evaluation')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for evaluation')
parser.add_argument('--model_path', default=None, type=str,
                    help='Checkpoint state_dict file to test from')
parser.add_argument('--result_path', default='./detection_output', type=str,
                    help='Path to store detection results in evaluation')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to evaluate model')
args = parser.parse_args()



if torch.cuda.is_available():
    print('CUDA devices: ', torch.cuda.device)
    print('GPU numbers: ', torch.cuda.device_count())
    num_gpus = torch.cuda.device_count()

num_gpus = 1


def evaluate(model: torch.nn.Module, forward_pass_callback_args):
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    Note: Honoring the number of iterations is not absolutely necessary.
    However if all evaluations run over an entire epoch of validation data,
    the runtime for AIMET compression will obviously be higher.

    :param model: Model to evaluate
    :param eval_iterations: Number of iterations to use for evaluation.
            None for entire epoch.
    :param use_cuda: If true, evaluate using gpu acceleration
    :return: single float number (accuracy) representing model's performance
    """
    # Assign imdb_name and imdbval_name according to args.dataset.
    if args.dataset == "voc":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
    # Import config
    if args.dataset == 'coco':
        cfg = (coco320, coco512)[args.input_size==512]
    elif args.dataset == 'voc': 
        cfg = (voc320, voc512)[args.input_size==512]
    # Create imdb, roidb and blob_dataset
    print('Create or load an evaluted imdb.')
    imdb, roidb = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))
    blob_dataset = BlobDataset(
        imdb, roidb, transform=BaseTransform(cfg['min_dim'], MEANS),
        target_normalization=True)

    '''
    # Construct networks.
    print('Construct {}_refinedet network.'.format(args.network))
    if args.network == 'vgg16':
        refinedet = VGGRefineDet(cfg['num_classes'], cfg)
    elif args.network == 'resnet101':
        refinedet = ResNetRefineDet(cfg['num_classes'], cfg)
    '''
    refinedet = model
    #refinedet.create_architecture()
    # For CPU
    net = refinedet
    # For GPU/GPUs
    if args.cuda:
        net = refinedet.cuda()
        if num_gpus > 1:
            net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # Load weights
    #net.load_weights(args.model_path)
    net.eval()
    print('Test RefineDet on:', args.imdbval_name)
    print('Using the specified args:')
    print(args)

    num_images = len(imdb.image_index)
    num_classes = imdb.num_classes
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    
    output_dir = get_output_dir(imdb, args.result_path)
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    # set no grad for variables
    torch.set_grad_enabled(False)
    for idx in range(num_images):
        img, gt, h, w = blob_dataset.pull_item(idx)
        input = Variable(img.unsqueeze(0))
        if args.cuda:
            input = input.cuda()
        # timers forward
        _t['im_detect'].tic()
        detection = net(input)
        detect_time = _t['im_detect'].toc(average=True)
        print('im_detect: {:d}/{:d} {:.3f}s'.format(
            idx + 1, num_images, detect_time))
        # skip jc = 0, because it's the background class
        for jc in range(1, num_classes):
            dets = detection[0, jc, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if (len(dets) > 0) and (dets.dim() > 0):
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[jc][idx] = cls_dets
            else:
                all_boxes[jc][idx] = empty_array
    
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
    

    
'''
def evaluate(model: torch.nn.Module, forward_pass_callback_args):
    """
     This is intended to be the user-defined model evaluation function. AIMET requires the above signature. So if the
     user's eval function does not match this signature, please create a simple wrapper.
     Use representative dataset that covers diversity in training data to compute optimal encodings.

    :param model: Model to evaluate
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
    """
    
    

    #print(model)
    #exit()
    model.eval()
    dummy_input = torch.randn(32, 3,320, 320).to(torch.device('cuda'))
    #refinedet320.to(torch.device('cpu'))
    with torch.no_grad():
        model(dummy_input)
'''   


def quantsim_refinedet():

    AimetLogger.set_level_for_all_areas(logging.INFO)
    
    cfg = (voc320, voc512)[0]
    refinedet320 = VGGRefineDet(cfg['num_classes'], cfg)
    refinedet320.create_architecture()
    
    refinedet320.load_state_dict(torch.load('/home/ava/sarala/RefineDet_PreTrained_Checkpoints/vgg16_refinedet320_voc_120000.pth'))
    refinedet320.eval()
    refinedet320.cuda()
    input_shape = (32, 3,320, 320)
    
    dummy_input = torch.randn(input_shape).cuda()

    # Prepare model for Quantization SIM. This will automate some changes required in model definition for example
    # create independent modules for torch.nn.functional and reused modules
    #prepared_model = prepare_model(refinedet320)

    equalize_model(refinedet320, input_shape)
    
    # Instantiate Quantization SIM. This will insert simulation nodes in the model
    quant_sim = QuantizationSimModel(refinedet320, dummy_input=dummy_input,
                                     quant_scheme=QuantScheme.post_training_tf_enhanced,
                                     default_param_bw=8, default_output_bw=8
                                     #,config_file='../../TrainingExtensions/common/src/python/aimet_common/quantsim_config/'
                                                 #'default_config.json'
                                                 )

    # Compute encodings (min, max, delta, offset) for activations and parameters. Use representative dataset
    # roughly ~1000 examples
    quant_sim.compute_encodings(evaluate, forward_pass_callback_args=None)
    

    # QAT - Quantization Aware Training - Fine-tune the model fore few epochs to retain accuracy using train loop
    #data_loader = create_fake_data_loader(dataset_size=32, batch_size=16, image_size=input_shape[1:])
    #_ = train(quant_sim.model, data_loader)

    # Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
    # activations and parameters in JSON format
    #quant_sim.export(path='./', filename_prefix='quantized_refinedet320', dummy_input=dummy_input.cpu())
    quant_sim.export(path='/home/ava/sarala/RefineDet/refinedet-onnxvalidation/pytorch_ptq/', filename_prefix='refinedet_sarala', dummy_input=dummy_input.cpu())
    
quantsim_refinedet()