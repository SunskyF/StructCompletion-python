import numpy as np
import cv2

from .trans_tform import trans_tform

eps = 1e-16

def debug(debuginfo):
    print(debuginfo)
    print(a)

def apply_tform_H(x, h7, h8):
    y = x[:, 0] * h7 + x[:, 1] * h8 + 1
    y = x[:, :2] / (y[..., None] + eps)
    return y

def src_domain_tform(uvPlaneID, modelPlane, modelReg, srcPos, trgPos, sampleRandReg):

    numUvPix = srcPos.shape[0]

    uvTformData = np.zeros((numUvPix, 9), dtype=np.float32)
    I = np.eye(3)

    for indPlane in range(modelPlane.numPlane):
        rectMat = modelPlane.rectMat[indPlane]
        h7 = rectMat[2, 0]
        h8 = rectMat[2, 1]

        uvPlaneIndCur = uvPlaneID == indPlane
        numPlanePixCur = np.sum(uvPlaneIndCur)

        if numPlanePixCur:

            trgPosCur = trgPos[uvPlaneIndCur, :].copy() - 1
            trgPosCurR = apply_tform_H(trgPosCur, h7, h8)

            if sampleRandReg:
                srcPosCur = srcPos[uvPlaneIndCur, :].copy() - 1
                srcPosCurR = apply_tform_H(srcPosCur, h7, h8)

                dRect = srcPosCurR - trgPosCurR

            else:
                dRect = np.zeros((numPlanePixCur, 2), dtype=np.float32)

                numDispVecCur = modelReg.numDispVec[indPlane]

                if numDispVecCur != 0:
                    randInd = np.random.randint(numDispVecCur, numPlanePixCur, 1)
                    dRect = modelReg.dispVec[indPlane][randInd, :]

            if dRect.shape[1] != 0:
                uvTformCur = np.zeros((numPlanePixCur, 9), dtype=np.float32)

                uvTformCur[:, np.array([0, 3, 6])] = dRect[:, 0][..., None] * np.array([h7, h8, 1])[None, ...]
                uvTformCur[:, np.array([1, 4, 7])] = dRect[:, 1][..., None] * np.array([h7, h8, 1])[None, ...]

                dTemp = dRect.dot([h7, h8])

                uvTformCur[:, np.array([2, 5, 8])] = dTemp[..., None] * -np.array([h7, h8, 1])[None, ...]

                uvTformCur = uvTformCur + I.reshape((1, 9))

                uvTformData[uvPlaneIndCur, :] = trans_tform(uvTformCur, trgPosCur)
                uvTformData[uvPlaneIndCur, 6:8] += 1

    return uvTformData