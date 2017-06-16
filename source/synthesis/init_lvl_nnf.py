from .prep_source_patch import prep_source_patch
from .prep_target_patch import prep_target_patch
from .patch_cost import patch_cost
from .voting import voting
from .src_domain_tform import src_domain_tform
from .update_uvMap import update_uvMap
from .prep_dist_patch import prep_dist_patch
from .clamp import clamp
from .uvMat_from_uvMap import uvMat_from_uvMap
from .check_valid_uv import check_valid_uv
from .trans_tform import trans_tform

import numpy as np
import cv2
from scipy import ndimage

def imshow(img):
    cv2.imshow("tmp", img)
    cv2.waitKey()

def getTargPathInd(NNF, option):
    pixIndMap = np.arange(NNF.imgH*NNF.imgW).reshape((NNF.imgW, NNF.imgH)).T[..., None].astype(np.float32)
    indTrgPatch = prep_target_patch(pixIndMap, NNF.uvPix.sub, option)

    trgPatchInd = indTrgPatch.reshape((option.pNumPix, NNF.uvPix.numPix))

    trgPatchInd = np.vstack([trgPatchInd, trgPatchInd+NNF.imgH*NNF.imgW, trgPatchInd+2*NNF.imgH*NNF.imgW])

    return trgPatchInd

def getUvPix(uvMap):
    uvPix = type("uvPix", (), {})

    uvPix.sub = np.array(np.where(uvMap.T)) # col, row
    uvPix.mask = uvMap
    uvPix.ind = np.ravel_multi_index([uvPix.sub[1], uvPix.sub[0]], uvMap.shape, order='F')
    uvPix.numPix = uvPix.ind.shape[0]

    return uvPix

def getUvPixN(NNF, option):
    uvPixN = []
    for i in range(4):
        singleUvPixN = type("singleUvPixN", (), {})
        singleUvPixN.sub = NNF.uvPix.sub - option.propDir[i, :][..., None] # 2 * N

        singleUvPixN.ind = np.ravel_multi_index([singleUvPixN.sub[1], singleUvPixN.sub[0]],
                                                [NNF.imgH, NNF.imgW], order='F')

        singleUvPixN.validInd = NNF.uvPix.mask[singleUvPixN.sub[1, :], singleUvPixN.sub[0, :]]

        uvPixN.append(singleUvPixN)

    return uvPixN

def prep_plane_prob_acc(planeProb, NNF):
    numPlane = planeProb.shape[2]
    numUvPix = NNF.uvPix.ind.shape[0]

    planeProbAcc = np.zeros((numUvPix, numPlane+1), dtype=np.float32)
    for i in range(numPlane):
        planeProbCur = planeProb[:, :, i]
        planeProbAcc[:, i+1] = planeProbCur[np.unravel_index(NNF.uvPix.ind, (NNF.imgH, NNF.imgW), order="F")]
        if i != 0:
            planeProbAcc[:, i+1] += planeProbAcc[:, i]

    return planeProbAcc

def init_level(mask, psize, prad):
    uvMask = cv2.dilate(mask, np.ones((psize, psize)))

    uvMask[:prad, :] = 0
    uvMask[-prad:, :] = 0
    uvMask[:, :prad] = 0
    uvMask[:, -prad:] = 0

    uvPix = getUvPix(uvMask)

    validMap = (1 - uvMask)
    validMap[:prad, :] = 0
    validMap[-prad:, :] = 0
    validMap[:, :prad] = 0
    validMap[:, -prad:] = 0

    validPix = getUvPix(validMap)

    return validPix, uvPix

def init_nnf(holeMask, modelPlane, modelReg, option):
    NNF = type("nnf", (), {})
    NNF.imgH, NNF.imgW = holeMask.shape
    NNF.validPix, NNF.uvPix = init_level(holeMask, option.pSize, option.pRad)

    NNF.uvPixN = getUvPixN(NNF, option)

    NNF.trgPatchInd = getTargPathInd(NNF, option)

    uvPlaneIDData = (modelPlane.numPlane - 1) * np.ones((NNF.uvPix.numPix), np.uint8)

    NNF.uvPlaneID = type("uvPId", (), {})
    NNF.uvPlaneID.data = uvPlaneIDData
    NNF.uvPlaneID.map = np.zeros((NNF.imgH, NNF.imgW), dtype=np.uint8)
    NNF.uvPlaneID.map[np.unravel_index(NNF.uvPix.ind, (NNF.imgH, NNF.imgW), order="F")] = NNF.uvPlaneID.data
    NNF.uvPlaneID.planeProbAcc = prep_plane_prob_acc(modelPlane.planeProb, NNF)
    NNF.uvPlaneID.mLogLikelihood = -np.log(NNF.uvPlaneID.planeProbAcc)

    # init uvTform
    if NNF.validPix.numPix:
        randInd = np.random.randint(NNF.validPix.numPix, size=(NNF.uvPix.numPix))
        uvRandSub = NNF.validPix.sub[:, randInd]
    else:
        uvRandSub = (NNF.imgW / 2) * np.ones((2, NNF.uvPix.numPix))

    NNF.uvTform = type("uvTform", (), {})
    NNF.uvTform.data = src_domain_tform(NNF.uvPlaneID.data, modelPlane, modelReg, uvRandSub, NNF.uvPix.sub, 1)
    NNF.uvTform.map = np.zeros((NNF.imgH, NNF.imgW, 9), dtype=np.float32)
    NNF.uvTform.map = update_uvMap(NNF.uvTform.map, NNF.uvTform.data, NNF.uvPix.ind)

    # init bias
    NNF.uvBias = type("uvBias", (), {})
    NNF.uvBias.data = np.zeros((1, 3, NNF.uvPix.numPix), dtype=np.float32)
    NNF.uvBias.map = np.zeros((NNF.imgH, NNF.imgW, 3), dtype=np.float32)

    # init cost
    NNF.uvCost = type("uvCost", (), {})
    NNF.uvCost.data = np.zeros((NNF.uvPix.numPix, 1), dtype=np.float32)
    NNF.uvCost.map = np.zeros((NNF.imgH, NNF.imgW), dtype=np.float32)

    # init distmap
    NNF.distMap, _ = ndimage.distance_transform_edt(1 - (holeMask == 0), return_indices=True)
    NNF.wPatchR, NNF.wPatchSumImg = prep_dist_patch(NNF.distMap, NNF.uvPix.sub, option)
    NNF.uvDtBdPixPos = NNF.distMap[np.unravel_index(NNF.uvPix.ind, (NNF.imgH, NNF.imgW), order="F")].astype(np.double)

    return NNF

def upsample(holeMask, NNF_L, modelPlane, modelReg, optS):
    NNF_H = type("nnf", (), {})
    NNF_H.imgH, NNF_H.imgW = holeMask.shape
    NNF_H.validPix, NNF_H.uvPix = init_level(holeMask, optS.pSize, optS.pRad)

    NNF_H.uvPixN = getUvPixN(NNF_H, optS)

    NNF_H.trgPatchInd = getTargPathInd(NNF_H, optS)

    imgH_H = NNF_H.imgH
    imgW_H = NNF_H.imgW
    imgH_L = NNF_L.imgH
    imgW_L = NNF_L.imgW

    sX = imgH_L / imgH_H
    sY = imgW_L / imgW_H

    uvPixL = type("uvPixL", (), {})
    uvPixL.sub = np.round(NNF_H.uvPix.sub.T.dot(np.diag([sX, sY])))
    uvPixL.sub[:, 1] = clamp(uvPixL.sub[:, 0], optS.pRad + 1, imgW_L - optS.pRad)
    uvPixL.sub[:, 2] = clamp(uvPixL.sub[:, 2], optS.pRad + 1, imgH_L - optS.pRad)
    uvPixL.ind = np.ravel_multi_index([uvPixL.sub[:, 1], uvPixL.sub[:, 0]], [imgH_L, imgW_L], order='F')

    NNF_H.uvPlaneID = type("uvPlaneID", (), {})
    NNF_H.uvPlaneID.data = uvMat_from_uvMap(NNF_L.uvPlaneID.map, uvPixL.ind)
    NNF_H.uvPlaneID.data[NNF_H.uvPlaneID.data == 0] = 1
    NNF_H.uvPlaneID.map = np.zeros((NNF_H.imgH, NNF_H.imgW), dtype=np.uint8)
    NNF_H.uvPlaneID.map = update_uvMap(NNF_H.uvPlaneID.map, NNF_H.uvPlaneID.data, NNF_H.uvPix.ind)

    NNF_H.uvPlaneID.planeProbAcc = prep_plane_prob_acc(modelPlane.planeProb, NNF_H)
    NNF_H.uvPlaneID.mLogLikelihood = -np.log(NNF_H.uvPlaneID.planeProbAcc)

    refineVec = NNF_H.uvPix.sub - uvPixL.sub * np.diag([1 / sX, 1 / sY])
    uvTform_H = trans_tform(uvTform_L, refineVec)

    uvTform_L = uvMat_from_uvMap(NNF_L.uvTform.map, uvPixL.ind)
    uvTform_L[:, 6: 8] = uvTform_L[:, 7: 8]*np.diag([1 / sX, 1 / sY])

    uvValid_H = check_valid_uv(uvTform_H[:, 7: 8], NNF_H.validPix.mask)

    uvInvalidInd = ~uvValid_H
    nInvalidUv_H = sum(uvInvalidInd)
    if nInvalidUv_H
        randInd = randi(size(NNF_H.validPix.ind, 1), nInvalidUv_H, 1)
        uvTform_H(uvInvalidInd, 7: 8) = NNF_H.validPix.sub(randInd,:)
        randInd = np.random.randint(NNF.validPix.numPix, size=(NNF.uvPix.numPix))
        uvRandSub = NNF.validPix.sub[:, randInd]


    NNF_H.distMap, _ = ndimage.distance_transform_edt(1 - (holeMask == 0), return_indices=True)
    NNF_H.wPatchR, NNF_H.wPatchSumImg = prep_dist_patch(NNF_H.distMap, NNF_H.uvPix.sub, optS)
    NNF_H.uvDtBdPixPos = NNF_H.distMap[np.unravel_index(NNF_H.uvPix.ind, (NNF_H.imgH, NNF_H.imgW), order="F")].astype(
        np.double)

    return NNF_H

def init_lvl_nnf(img, NNF, holeMask, modelPlaneCur, modelRegCur, option):

    if option.iLvl == option.numPyrLvl - 1:
        NNF = init_nnf(holeMask, modelPlaneCur, modelRegCur, option)
    else:
        NNF = upsample(holeMask, NNF, modelPlaneCur, modelRegCur, option)

        trgPatch = prep_target_patch(img, NNF.uvPix.sub, option)
        srcPatch = prep_source_patch(img, NNF.uvTform.data, option)

        _, NNF.uvBias.data = patch_cost(trgPatch, srcPatch, modelPlaneCur, NNF.uvPlaneID.data,
                                NNF.uvPix.sub, NNF.uvTform.data, NNF.uvTform.map[:, :, 6:8], NNF.uvDtBdPixPos, option)

        img = voting(img, NNF, holeMask, option)

    return img, NNF