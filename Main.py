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
    RESNET152 = 'RES-NET-152'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    #---- Path to the directory with images
    #pathDirData = '/home/nthieuitus/bse_chestxray'
    #pathDirData = '/home/nthieuitus/chestxray'
    #pathDirData = '/home/nthieuitus/inverted_bse_chestxray'
    #pathDirData = '/home/nthieuitus/bse_equalization'
    #pathDirData = '/home/nthieuitus/bse_equal'
    pathDirData = '/home/minhchuong_itus/bse_chestxray'

    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'

    #---- Neural network parameters: type of the network, is it pre-trained
    #---- on imagenet, number of classes
    nnArchitecture = RESNET152
    nnIsTrained = True
    nnClassCount = 14

    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 200
    trMaxEpoch = 100

    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'm-' + timestampLaunch + '.pth.tar'

    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)

    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch, "predict_bse_CLAHE.txt")

#--------------------------------------------------------------------------------

def runTest():

    pathDirData = '/home/nthieuitus/bse_chestxray'
    pathDirData_origin = '/home/nthieuitus/chestxray'
    pathFileTest = './dataset/test_2.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 230
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'S3.pth.tar'
    pathModel_origin = 'W.pth.tar'

    timestampLaunch = ''

    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch, "predict_bse_CLAHE.txt")
    #ChexnetTrainer.test(pathDirData_origin, pathFileTest, pathModel_origin, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch, "predict_origin.csv")

#--------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
