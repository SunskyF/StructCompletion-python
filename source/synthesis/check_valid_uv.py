import numpy as np

def check_valid_uv(srcPos, validSrcMask):
    srcPos_in = np.round(srcPos)
    numUvPix = srcPos.shape[0]

    uvValidInd = np.zeros((numUvPix, 1))

    validSrcInd = (srcPos_in[:, 0] >= 0) & (srcPos_in[:, 0] < validSrcMask.shape[1]) \
                    & (srcPos_in[:, 1] >= 0) & (srcPos_in[:, 1] < validSrcMask.shape[0])

    uvValidInd[validSrcInd] = validSrcMask[srcPos_in[validSrcInd, 1:2].astype(np.int), srcPos_in[validSrcInd, 0:1].astype(np.int)]
    return uvValidInd