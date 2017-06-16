import numpy as np

from .prep_target_patch import prep_target_patch

def prep_dist_patch(distMap, trgPixPos, option):
    imgH, imgW = distMap.shape
    wPatchR = prep_target_patch(distMap[..., None], trgPixPos, option)
    wPatchR = np.squeeze(wPatchR)

    wPatchR = wPatchR - wPatchR[option.pMidPix, :]
    wPatchR = option.wDist[option.iLvl] ** wPatchR

    numUvPix = wPatchR.shape[1]

    wPatchSumImg = np.zeros((imgH, imgW), dtype=np.float32)
    indMap = np.reshape(range(imgH * imgW), (imgH, imgW), order="F")

    indPatch = prep_target_patch(indMap[..., None], trgPixPos, option)
    indPatch = np.squeeze(indPatch).astype(np.int)

    for i in range(numUvPix):
        wPatchSumImg[np.unravel_index(indPatch[:, i], (imgH, imgW), order="F")] += wPatchR[:, i]

    return wPatchR, wPatchSumImg