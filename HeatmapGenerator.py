import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

#--------------------------------------------------------------------------------
#---- Class to generate heatmaps (CAM)
evaluated = {}

images = {}
ratio = 1.0#224.0/1024
with open("heatmap/bbox.txt", "r") as f:
    for line in f.readlines():
        elements = line.strip().split(',')
        if len(elements) > 2:
            images[elements[0]] = {'label': elements[1], \
                                    'x': float(elements[2])*ratio, \
                                    'y': float(elements[3])*ratio,
                                    'w': float(elements[4])*ratio, \
                                    'h': float(elements[5])*ratio}

class HeatmapGenerator ():

    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14


    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):

        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True)#.cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True)#.cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True)#.cuda()

        model = torch.nn.DataParallel(model)#.cuda()

        modelCheckpoint = torch.load(pathModel, map_location='cpu')
        model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model.module.densenet121.features
        self.model.eval()

        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)

        self.transformSequence = transforms.Compose(transformList)

    #--------------------------------------------------------------------------------

    def generate (self, pathImageFile, pathOutputFile, transCrop, isBSE=False):

        #---- Load image, transform, convert
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)

        input = torch.autograd.Variable(imageData)

        #self.model.cuda()
        output = self.model(input)#.cuda())

        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map

        #---- Blend original and heatmap
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        try:
            imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        except:
            print("Error on file ", pathImageFile)
            return
        cam = npHeatmap / np.max(npHeatmap)

        #Storing heatmap value
        newCam = cv2.resize(cam, (1024,1024))
        file = pathImageFile.split('/')[-1]
        image_detail = images[file]
        x = int(image_detail['x'])
        y = int(image_detail['y'])
        w = int(image_detail['w'])
        h = int(image_detail['h'])
        if not isBSE:
            print("PROCESS BSE")
            evaluated[file] = (np.mean(newCam[y:y+h, x:x+w]),)
        else:
            print("PROCESS origin")
            value = evaluated.get(file, (0,))
            evaluated[file] = (value[0], np.mean(newCam[y:y+h, x:x+w]))

        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

        img = heatmap * 0.5 + imgOriginal

        #cv2.imwrite(pathOutputFile, img)

#--------------------------------------------------------------------------------

pathInputImage = 'test/00009285_000.png'
pathOutputImage = 'test/heatmap_1.png'
pathModel = 'models/W.pth.tar'

nnArchitecture = 'DENSE-NET-121'
nnClassCount = 14

transCrop = 224

origin_input = "/Volumes/Ryan/CheXNet/test_visualization/origin_choosing_bse"
origin_output = "/Volumes/Ryan/CheXNet/test_visualization/origin_choosing_bse_heatmap"
bse_input = "/Volumes/Ryan/CheXNet/test_visualization/bse_choosing_bse"
bse_output = "/Volumes/Ryan/CheXNet/test_visualization/bse_choosing_bse_heatmap"

h = HeatmapGenerator('models/W.pth.tar', nnArchitecture, nnClassCount, transCrop)
for file in os.listdir(origin_input):
    if not file.endswith(".png"):
        continue
    pathInputImage = os.path.join(origin_input, file)
    pathOutputImage = os.path.join(origin_output, file)
    h.generate(pathInputImage, pathOutputImage, transCrop, isBSE=False)

h = HeatmapGenerator('models/CLAHE-0152.pth.tar', nnArchitecture, nnClassCount, transCrop)
for file in os.listdir(bse_input):
    if not file.endswith(".png"):
        continue
    pathInputImage = os.path.join(bse_input, file)
    pathOutputImage = os.path.join(bse_output, file)
    h.generate(pathInputImage, pathOutputImage, transCrop, isBSE=True)

evaluateFile = open("new_evaluate.csv", "w")
for key, value in evaluated.items():
    if len(value) < 2:
        continue
    result = 'origin' if value[1] - value[0] >= 50 else 'bse'
    evaluateFile.write("{},{},{},{}\n".format(key, value[0], value[1], result))

evaluateFile.close()
