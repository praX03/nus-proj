import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import sys
import os

from torch.autograd import Variable
cuda_available = torch.cuda.is_available()
print("Checking for GPU on device:")
print("Status: ", cuda_available)
parser = argparse.ArgumentParser(description='image encoder')
parser.add_argument('--img_data', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()
output_dir = args.output
image_dir = args.img_data

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if cuda_available:
    net = models.vgg19_bn().cuda()
else:
    net = models.vgg19_bn().cpu()
net.classifier = nn.Sequential(*list(net.classifier.children())[:-6])

print('---------loading decode training data---------')
img_data = torchvision.datasets.ImageFolder(image_dir, transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()]))
testdata = torch.utils.data.DataLoader(img_data)
features = {}
image = {}

for step, (x, y) in enumerate(testdata):
    if cuda_available:
        b_x = x.cuda()
    else:
        b_x = x.cpu()
    data = net(b_x)
    xxx = data.cpu().data.numpy()
    features[os.path.basename(img_data.imgs[step][0])] = xxx.flatten()
    image[os.path.basename(img_data.imgs[step][0])] = x.numpy()

file1 = open(os.path.join(output_dir,'images_vgg19_bn_fc.pickle'), 'wb')
pickle.dump(image, file1)
file1.close()

file2 = open(os.path.join(output_dir,'features_vgg19_bn_fc.pickle'), 'wb')
pickle.dump(features, file2)
file2.close()
print('---------decode training data saved---------')