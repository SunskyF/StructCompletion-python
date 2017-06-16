import cv2
import numpy as np
import os
import math
import time

from .init_lvl_nnf import init_lvl_nnf
from .sc_pass import sc_pass

def synthesis(imgPyr, maskPyr, modelPlane, modelReg, option):
    NNF = []
    numIterLvl = option.numIter

    for iLvl in range(option.numPyrLvl-1, -1, -1):
        holeMask = maskPyr[iLvl]
        H, W = holeMask.shape

        option.iLvl = iLvl
        option.imgSize = max(H, W)

        modelPlaneCur = modelPlane[iLvl]
        modelRegCur = modelReg[iLvl]

        print("--- Initialize NNF: ")
        time0 = time.time()
        img, NNF = init_lvl_nnf(imgPyr[iLvl], NNF, holeMask, modelPlaneCur, modelRegCur, option)
        print('Done in %6.3f seconds.\n' % (time.time() - time0))

        numIterLvl = int(max(numIterLvl - option.numIterDec, option.numIterMin))
        option.numIterLvl = numIterLvl

        print("--- Pass... level: {}, #Iter: {}, #uvPixels: {}".format(iLvl, numIterLvl, NNF.uvPix.numPix))
        print("--- #iter\t#PropUpdate\t\t#RandUpdate\t\t#RegUpdate\t\tAvgCost")
        print(option.iLvl)
        print(option.numPyrLvl - 1)
        if option.iLvl == option.numPyrLvl - 1:
            img, NNF = sc_pass(img, holeMask, NNF, modelPlaneCur, modelRegCur, option, 1)

            img, NNF = sc_pass(img, holeMask, NNF, modelPlaneCur, modelRegCur, option, 0)
        else:
            img, NNF = sc_pass(img, holeMask, NNF, modelPlaneCur, modelRegCur, option, 0)

        imgPyr[iLvl] = img
    return imgPyr