import torch
import torchvision

from libs.networks.vgg_refinedet import VGGRefineDet
from libs.utils.config import voc320, voc512


cfg = (voc320, voc512)[1]
refinedet512 = VGGRefineDet(cfg['num_classes'], cfg)
refinedet512.create_architecture()

refinedet512.load_state_dict(torch.load('./output/vgg16_refinedet512_voc_120000.pth'))


#input_value = (batch_size, height, width, channel)
input_names = ["input"]
output_names = ["output"]


#dummy input defined with batch size = 32 and  image dimensions 320, 320, 3
dummy_input = torch.randn(32, 3, 512, 512)


torch.onnx.export(refinedet512, dummy_input, "/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/refinedet512.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=11)