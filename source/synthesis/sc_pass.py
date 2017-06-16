from .prep_source_patch import prep_source_patch
from .prep_target_patch import prep_target_patch
from .patch_cost import patch_cost
from .update_uvMap import update_uvMap
from .voting import voting
from .update_NNF import update_NNF

import cv2
import numpy as np
import os
import math

def imshow(img):
    cv2.namedWindow("tmp", cv2.WINDOW_NORMAL)
    cv2.imshow("tmp", img)

def sc_pass(img, holeMask, NNF, modelPlaneCur, modelRegCur, option, lockImgFlag):

    for iter in range(option.numIterLvl):
        trgPatch = prep_target_patch(img.copy(), NNF.uvPix.sub, option)
        srcPatch = prep_source_patch(img.copy(), NNF.uvTform.data, option)

        uvCostcur, NNF.uvBias.data = patch_cost(trgPatch, srcPatch, modelPlaneCur, NNF.uvPlaneID.data,
                   NNF.uvPix.sub, NNF.uvTform.data, NNF.uvTform.map[:, :, 6:8],
                   NNF.uvDtBdPixPos, option)

        NNF.uvCost.data = np.sum(uvCostcur, axis=1, keepdims=True)

        if len(NNF.uvCost.map.shape) == 2:
            NNF.uvCost.map = NNF.uvCost.map[..., None]
        NNF.uvCost.map = update_uvMap(NNF.uvCost.map, NNF.uvCost.data, NNF.uvPix.ind)

        NNF, nUpdate = update_NNF(trgPatch, img, NNF, modelPlaneCur, modelRegCur, option)
        avgPatchCost = np.mean(NNF.uvCost.data, axis=0)

        if not lockImgFlag:
            img = voting(img, NNF, holeMask, option)
            imshow(img)

        print("--- {}\t\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}".format(
            iter, nUpdate[0], nUpdate[1], nUpdate[2], avgPatchCost[0]))

    if option.useBiasCorrection:
        uvBias = np.reshape(NNF.uvBias.data, (NNF.uvPix.numPix, 3))

        NNF.uvBias.map = update_uvMap(NNF.uvBias.map, uvBias, NNF.uvPix.ind)
    return img, NNF