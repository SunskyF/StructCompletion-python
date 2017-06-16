import numpy as np

def check_valid_uv(srcPos, validSrcMask):
    srcPos = np.round(srcPos)
    numUvPix = srcPos.shape[0]

    uvValidInd = np.zeros((numUvPix, 1))

    validSrcInd = (srcPos[:, 0] >= 0) & (srcPos[:, 0] < validSrcMask.shape[1]) \
                    & (srcPos[:, 1] >= 0) & (srcPos[:, 1] < validSrcMask.shape[0])

    uvValidInd[validSrcInd] = validSrcMask[srcPos[validSrcInd, 1:2].astype(np.int), srcPos[validSrcInd, 0:1].astype(np.int)]
    return uvValidInd