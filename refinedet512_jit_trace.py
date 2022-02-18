import torch
import torchvision

from libs.networks.vgg_refinedet import VGGRefineDet
from libs.utils.config import voc320, voc512


cfg = (voc320, voc512)[1]
refinedet512 = VGGRefineDet(cfg['num_classes'], cfg)
refinedet512.create_architecture()

#refinedet320.load_state_dict(torch.load('./output/vgg16_refinedet320_voc_120000.pth'))
# Trace a module (implicitly traces `forward`) and construct a
# `ScriptModule` with a single `forward` method
example_forward_input = torch.randn(32, 3, 512, 512)
#module = torch.jit.trace(refinedet512.forward, example_forward_input)
module = torch.jit.trace(refinedet512, example_forward_input)
