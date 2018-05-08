import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

#--------------------------------------------------------------------------------

def main ():

    #runTest()
    runTrain()

#--------------------------------------------------------------------------------

def runTrain():

    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    #---- Path to the directory with images
    pathDirData = '/home/minhchuong_itus/bse_chestxray'
    #pathDirData = '/home/minhchuong_itus/chexnet/CheXNet/ChestX-ray14'

    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train.txt'
    pathFileVal = './dataset/dev.txt'
    pathFileTest = './dataset/test.txt'

    #---- Neural network parameters: type of the network, is it pre-trained
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 14

    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 128
    trMaxEpoch = 100

    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'models/m-' + timestampLaunch + '.pth.tar'

    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)

    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#--------------------------------------------------------------------------------

def runTest():

    pathDirData = '/home/minhchuong_itus/bse_chestxray'
    pathFileTest = './dataset/test.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 200
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'm-07052018-150224.pth.tar'

    timestampLaunch = ''

    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#--------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
