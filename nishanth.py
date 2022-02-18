import argparse
import copy
import csv
import os

import torch
import tqdm
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

import aimet_torch
import aimet_common
import numpy as np
from aimet_torch import bias_correction
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams, QuantizationSimModel 
from aimet_common.defs import QuantScheme
from nets import nn
from utils import util

data_dir = "/home/ava/Imagenet/imagenet-1k/"

def batch(images, target, model, criterion=None):
    images = images.cuda()
    target = target.cuda()
    if criterion:
        with torch.cuda.amp.autocast():
            loss = criterion(model(images), target)
        return loss
    else:
        return util.accuracy(model(images), target, top_k=(1, 5))

def train(args):
    epochs = 350
    batch_size = 256
    util.set_seeds(args.rank)
    model = nn.EfficientNet(args).cuda()
    lr = batch_size * torch.cuda.device_count() * 0.256 / 4096
    optimizer = nn.RMSprop(util.add_weight_decay(model), lr, 0.9, 1e-3, momentum=0.9)
    ema = nn.EMA(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = nn.StepLR(optimizer)
    amp_scale = torch.cuda.amp.GradScaler()

    if args.tf:
        last_name = 'last_tf'
        best_name = 'best_tf'
        step_name = 'step_tf'
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        last_name = 'last_pt'
        best_name = 'best_pt'
        step_name = 'step_pt'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                   transforms.Compose([util.RandomResize(),
                                                       transforms.ColorJitter(0.4, 0.4, 0.4),
                                                       transforms.RandomHorizontalFlip(),
                                                       util.RandomAugment(),
                                                       transforms.ToTensor(), normalize]))
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    loader = data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    with open(f'weights/{step_name}.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'acc@1', 'acc@5'])
            writer.writeheader()
        best_acc1 = 0
        for epoch in range(0, epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                bar = tqdm.tqdm(loader, total=len(loader))
            else:
                bar = loader
            model.train()
            for images, target in bar:
                loss = batch(images, target, model, criterion)
                optimizer.zero_grad()
                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                ema.update(model)
                torch.cuda.synchronize()
                if args.local_rank == 0:
                    bar.set_description(('%10s' + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), loss))

            scheduler.step(epoch + 1)
            if args.local_rank == 0:
                acc1, acc5 = test(args, ema.model.eval())
                writer.writerow({'acc@1': str(f'{acc1:.3f}'),
                                 'acc@5': str(f'{acc5:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                state = {'model': copy.deepcopy(ema.model).half()}
                torch.save(state, f'weights/{last_name}.pt')
                if acc1 > best_acc1:
                    torch.save(state, f'weights/{best_name}.pt')
                del state
                best_acc1 = max(acc1, best_acc1)
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(model,  eval_iterations: int):
    '''if model is None:
            model = torch.load('weights/best_pt.pt', map_location='cuda')['model'].float().eval()
            '''

    '''
    input_shape = (1, 3, 384, 384)
    dummy_input = torch.rand(input_shape).cuda()
    quantsim = QuantizationSimModel(model=model, quant_scheme='tf',
                                    dummy_input=dummy_input, rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8)

    quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),
                               forward_pass_callback_args=iterations)
    

    quantsim.export(path=logdir, filename_prefix='resnet_encodings', dummy_input=dummy_input.cpu())
    accuracy = evaluator(quantsim.model, use_cuda=use_cuda)
    '''

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                   transforms.Compose([transforms.Resize(416),
                                                       transforms.CenterCrop(384),
                                                       transforms.ToTensor(), normalize]))

    loader = data.DataLoader(dataset, 48, num_workers=os.cpu_count(), pin_memory=True)
    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    with torch.no_grad():
        for images, target in tqdm.tqdm(loader, ('%10s' * 2) % ('acc@1', 'acc@5')):
            acc1, acc5 = batch(images, target, model)
            torch.cuda.synchronize()
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        acc1, acc5 = top1.avg, top5.avg
        print('%10.3g' * 2 % (acc1, acc5))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return acc1, acc5

model = torch.load('weights/best_pt.pt', map_location='cuda')['model'].float().eval()
input_shape = (1, 3, 384, 384)

# modelScript = torch.jit.trace(model.to(torch.device('cpu')), torch.tensor(np.random.rand(1, 3, 384, 384).astype(np.float32)))
# print("#"*100)

dummy_input = torch.rand(input_shape).cuda()
equalize_model(model, input_shape)

# quant_scheme = QuantScheme.post_training_tf_enhanced
quantsim = QuantizationSimModel(model=model, quant_scheme="tf_enhanced",
                                dummy_input=dummy_input, rounding_mode='nearest',
                                default_output_bw=16, default_param_bw=16)

#quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),forward_pass_callback_args=iterations)
quantsim.compute_encodings(forward_pass_callback=test, forward_pass_callback_args=5)

#print(quantsim)
#quantsim.export(path='./', filename_prefix='effnet_v2-S', dummy_input=input_shape)
#quantsim.export(path=logdir, filename_prefix='effnet_v2S', dummy_input=dummy_input.cpu())

accuracy = test(quantsim.model, 1)
quantsim.export(path='/home/ava/anaconda3/envs/efficientnetv2_pytorch/nishanth_efficientnetv2_pytorch/EffcientNetV2/nishanth_16_16_tf_enhanced/', filename_prefix='effnet_v2-S_nishanth', dummy_input=torch.rand(input_shape, device="cpu"))
                # onnx_export_args=(aimet_torch.onnx_utils.OnnxExportApiArgs (opset_version=11)))
                
''' 
changed the line inside the torch library
     anaconda3\envs\efficientnetv2pytorch\Lib\site-packages\torch\nn\functional.py
     In Line number: 1742
      replace "return torch._C._nn.silu(input)" => "return input*torch.sigmoid(input)"
'''

def print_parameters(args):
    model = nn.EfficientNet(args).eval()
    _ = model(torch.zeros(1, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {int(params)}')

def benchmark(args):
    shape = (1, 3, 384, 384)
    util.torch2onnx(nn.EfficientNet(args).export().eval(), shape)
    util.onnx2caffe()
    util.print_benchmark(shape)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tf', action='store_true')

    args = parser.parse_args()
    args.distributed = False
    args.rank = 0
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.rank = torch.distributed.get_rank()
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    if args.local_rank == 0:
        print_parameters(args)
    if args.benchmark:
        benchmark(args)
    if args.train:
        train(args)
    if args.test:
        test(args)

if __name__ == '__main__':
    main()
