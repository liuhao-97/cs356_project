import pytorch_nndct
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import optim
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10
import torch.nn.init as init

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--pretrained',
    default='/workspace/resnet20_cifar10/resnet20-200-regular.pth',
    help='Pre-trained model file path.')
parser.add_argument(
    '--workers',
    default=4,
    type=int,
    help='Number of data loading workers to be used.')
parser.add_argument('--epochs', default=3, type=int, help='Training epochs.')
parser.add_argument(
    '--quantizer_lr',
    default=1e-2,
    type=float,
    help='Initial learning rate of quantizer.')
parser.add_argument(
    '--quantizer_lr_decay',
    default=0.5,
    type=int,
    help='Learning rate decay ratio of quantizer.')
parser.add_argument(
    '--weight_lr',
    default=1e-5,
    type=float,
    help='Initial learning rate of network weights.')
parser.add_argument(
    '--weight_lr_decay',
    default=0.94,
    type=int,
    help='Learning rate decay ratio of network weights.')
parser.add_argument(
    '--train_batch_size', default=24, type=int, help='Batch size for training.')
parser.add_argument(
    '--val_batch_size',
    default=100,
    type=int,
    help='Batch size for validation.')
parser.add_argument(
    '--weight_decay', default=1e-4, type=float, help='Weight decay.')
parser.add_argument(
    '--display_freq',
    default=100,
    type=int,
    help='Display training metrics every n steps.')
parser.add_argument(
    '--val_freq', default=1000, type=int, help='Validate model every n steps.')
parser.add_argument(
    '--quantizer_norm',
    default=True,
    type=bool,
    help='Use normlization for quantizer.')
parser.add_argument(
    '--mode',
    default='train',
    choices=['train', 'deploy'],
    help='Running mode.')
parser.add_argument(
    '--save_dir',
    default='./qat_models',
    help='Directory to save trained models.')
parser.add_argument(
    '--output_dir', default='qat_result', help='Directory to save qat result.')
args, _ = parser.parse_known_args()

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(
            in_planes, planes, stride=stride, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(
            planes, planes,  stride=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.skip_add = functional.Add()
        self.skipexist = False
        

        if stride != 1 or in_planes != planes:
            self.skipexist = True
            self.conv3 = conv1x1(
                in_planes, self.expansion * planes,  stride=stride)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skipexist == True:
            identity = self.conv3(x)
            identity = self.bn3(identity)

        out = self.skip_add(out, identity)
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16, 
                                stride=1, dilation=1 )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(_weights_init)

        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_stub(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        # x = F.avg_pool2d(x, x.size()[3])

        # x = x.view(x.size(0), -1)
        # print(x.size())
        x = torch.flatten(x, 1)
        x = self.linear(x)

        x = self.dequant_stub(x)
        return x


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

    
def train_one_step(model, inputs, criterion, optimizer, step, gpu=None):
    # switch to train mode
    model.train()

    images, target = inputs

    if gpu is not None:
        model = model.cuda(gpu)
        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

    # compute output
    output = model(images)
    loss = criterion(output, target)

    l2_decay = 1e-4
    l2_norm = 0.0
    for param in model.quantizer_parameters():
        l2_norm += torch.pow(param, 2.0)[0]
    if args.quantizer_norm:
        loss += l2_decay * torch.sqrt(l2_norm)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc1, acc5


def validate(val_loader, model, criterion, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                model = model.cuda(gpu)
                images = images.cuda(gpu, non_blocking=True)
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

    return top1.avg


def mkdir_if_not_exist(x):
    if not x or os.path.isdir(x):
        return
    os.mkdir(x)
    if not os.path.isdir(x):
        raise RuntimeError("Failed to create dir %r" % x)


def save_checkpoint(state, is_best, directory):
    mkdir_if_not_exist(directory)

    filepath = os.path.join(directory, 'model.pth')
    torch.save(state, filepath)
    if is_best:
        best_acc1 = state['best_acc1'].item()
        best_filepath = os.path.join(
            directory, 'model_best_%5.3f.pth' % best_acc1)
        shutil.copyfile(filepath, best_filepath)
        print('Saving best ckpt to {}, acc1: {}'.format(best_filepath, best_acc1))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, step):
    """Sets the learning rate to the initial LR decayed by decay ratios"""

    weight_lr_decay_steps = 3000 * (24 / args.train_batch_size)
    quantizer_lr_decay_steps = 1000 * (24 / args.train_batch_size)

    for param_group in optimizer.param_groups:
        group_name = param_group['name']
        if group_name == 'weight' and step % weight_lr_decay_steps == 0:
            lr = args.weight_lr * (
                args.weight_lr_decay**(step / weight_lr_decay_steps))
            param_group['lr'] = lr
            print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
                epoch, step, group_name, lr))
        if group_name == 'quantizer' and step % quantizer_lr_decay_steps == 0:
            lr = args.quantizer_lr * (
                args.quantizer_lr_decay**(step / quantizer_lr_decay_steps))
            param_group['lr'] = lr
            print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
                epoch, step, group_name, lr))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, train_loader, val_loader, criterion, gpu):
    best_acc1 = 0
    num_train_batches_per_epoch = int(
        len(train_loader) / args.train_batch_size)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    param_groups = [{
        'params': model.quantizer_parameters(),
        'lr': args.quantizer_lr,
        'name': 'quantizer'
    }, {
        'params': model.non_quantizer_parameters(),
        'lr': args.weight_lr,
        'name': 'weight'
    }]
    optimizer = torch.optim.Adam(
        param_groups, args.weight_lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        progress = ProgressMeter(
            len(train_loader) * args.epochs,
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch[{}], Step: ".format(epoch))

        for i, (images, target) in enumerate(train_loader):
            end = time.time()
            # measure data loading time
            data_time.update(time.time() - end)

            step = len(train_loader) * epoch + i

            adjust_learning_rate(optimizer, epoch, step)
            loss, acc1, acc5 = train_one_step(model, (images, target), criterion,
                                              optimizer, step, gpu)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if step % args.display_freq == 0:
                progress.display(step)

            if step % args.val_freq == 0:
                # evaluate on validation set
                acc1 = validate(val_loader, model, criterion, gpu)

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1
                    }, is_best, args.save_dir)


def evaluate(net, test_data):
    net.eval()
    with torch.no_grad():
        correct = 0
        test_num = test_data.dataset.data.shape[0]
        total_test_loss = []

        for i, data in enumerate(test_data):

            inputs, labels = data  # [100,3,32,32]
            # print(len(inputs[0]))

            # inputs = inputs.cuda()
            # labels = labels.cuda()
            outputs = net(inputs)
            correct += torch.sum(torch.argmax(outputs, dim=1) == labels)

    return float(correct)/test_num

# def gen_calib( test_data):
#     with torch.no_grad():
#         for i, data in enumerate(test_data):
#             inputs, labels = data #[100,3,32,32]
#             # print(len(inputs[0]))
#             inputs_calib = inputs


#     return inputs_calib
if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader= torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)


    

    test_dataset = CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    

    model = resnet20()
    model.load_state_dict(torch.load('resnet20-200-regular.pth'))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    gpu = 0
    inputs = torch.randn([args.train_batch_size, 3, 32, 32],
                         dtype=torch.float32).cuda(gpu)

    qat_processor = QatProcessor(
        model, inputs, bitwidth=4, device=torch.device('cuda:{}'.format(gpu)))

    if args.mode == 'train':
        # Step 1: Get quantized model and train it.
        quantized_model = qat_processor.trainable_model()

        criterion = criterion.cuda(gpu)
        train(quantized_model, train_loader, test_loader, criterion, gpu)

        # Step 2: Get deployable model and test it.
        # There may be some slight differences in accuracy with the quantized model.
        deployable_model = qat_processor.to_deployable(quantized_model,
                                                       args.output_dir)
        validate(test_loader, deployable_model, criterion, gpu)
    elif args.mode == 'deploy':
        # Step 3: Export xmodel from deployable model.
        deployable_model = qat_processor.deployable_model(
            args.output_dir, used_for_xmodel=True)
        test_subset = torch.utils.data.Subset(test_dataset, list(range(1)))
        subset_loader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
        # Must forward deployable model at least 1 iteration with batch_size=1
        for images, _ in subset_loader:
            deployable_model(images)
        qat_processor.export_xmodel(args.output_dir)
        print('aaaaa')
    else:
        raise ValueError('mode must be one of ["train", "deploy"]')
