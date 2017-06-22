import numpy as np
import cv2

from .prep_source_patch import prep_source_patch

def clamp(v, low, high):
    v[v < low] = low
    v[v > high] = high
    return v

def voting(img, NNF, holeMask, option):
    numUvPix = NNF.uvPix.numPix
    H, W, Ch = img.shape

    srcPatch = prep_source_patch(img, NNF.uvTform.data, option)

    if not option.useBiasCorrection:
        biasPatch = NNF.uvBias.data
        srcPatch = srcPatch + biasPatch

    trgPatchInd = NNF.trgPatchInd

    wPatchR = np.reshape(NNF.wPatchR, (option.pNumPix, 1, numUvPix))
    srcPatch = srcPatch * wPatchR

    srcPatch = np.reshape(srcPatch, (option.pNumPix * Ch, numUvPix))

    imgAcc = np.zeros((H, W, Ch), dtype=np.float32)
    for i in range(numUvPix):
        imgAcc[np.unravel_index(trgPatchInd[:, i].astype(np.int), imgAcc.shape, order="F")] += srcPatch[:, i]

    imgAcc /= NNF.wPatchSumImg[..., None]

    img[holeMask == 1, :] = imgAcc[holeMask == 1, :]

    img = clamp(img, 0, 1)
    return img