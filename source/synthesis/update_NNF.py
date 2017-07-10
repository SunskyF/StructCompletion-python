import numpy as np
import cv2

from .trans_tform import trans_tform
from .uvMat_from_uvMap import uvMat_from_uvMap
from .check_valid_uv import check_valid_uv
from .prep_source_patch import prep_source_patch
from .patch_cost import patch_cost
from .update_uvMap import update_uvMap
from .src_domain_tform import src_domain_tform
from .draw_plane_id import draw_plane_id
from .scale_tform import scale_tform

def debug(debug_info):
    print(debug_info)
    print(a)

def shape(a):
    print(a.shape)
    print(b)

def propagate(trgPatch, img, NNF, modelPlane, option, iDirect):
    nUpdateTotal = 0
    uvPixN = NNF.uvPixN[iDirect]
    uvPixActiveInd = uvPixN.validInd.astype(np.int)

    numUpdatePix = NNF.uvPix.numPix

    while numUpdatePix != 0:
        uvPix = type("uvPix", (), {})
        uvPix.sub = NNF.uvPix.sub[uvPixActiveInd == 1, :].copy()
        uvPix.ind = NNF.uvPix.ind[uvPixActiveInd == 1].copy()

        uvPixNCur = type("uvPixNCur", (), {})
        uvPixNCur.sub = uvPixN.sub[uvPixActiveInd == 1, :].copy()
        uvPixNCur.ind = uvPixN.ind[uvPixActiveInd == 1].copy()

        uvDtBdPixPosCur = NNF.uvDtBdPixPos[uvPixActiveInd == 1].copy()

        trgPatchCur = trgPatch[:, :, uvPixActiveInd == 1].copy()
        srcPosCur = NNF.uvTform.data[uvPixActiveInd == 1, 6:8].copy()
        uvCostCur = NNF.uvCost.data[uvPixActiveInd == 1].copy()
        uvPlaneIDCur = NNF.uvPlaneID.data[uvPixActiveInd == 1].copy()

        srcPosMapCur = NNF.uvTform.map[:, :, 6:8].copy()
        uvPixActivePos = np.where(uvPixActiveInd == 1)[0]

        uvTformCand = uvMat_from_uvMap(NNF.uvTform.map, uvPixNCur.ind)
        uvTformCand = trans_tform(uvTformCand, option.propDir[iDirect, :])

        uvValidSrcInd = check_valid_uv(uvTformCand[:, 6:8], NNF.validPix.mask)

        diff = np.abs(uvTformCand[:, 6:8] - srcPosCur)
        uvValidDistInd = (diff[:, 0:1] > 1) | (diff[:, 1:2] > 1)

        uvValidInd = (uvValidSrcInd == 1) & (uvValidDistInd == 1)

        numUvValid = np.sum(uvValidInd)

        if numUvValid != 0:
            uvPixValid = type("uvPixValid", (), {})
            uvPixValid.sub = uvPix.sub[uvValidInd.squeeze() == 1, :].copy()
            uvPixValid.ind = uvPix.ind[uvValidInd.squeeze()].copy()

            uvDtBdPixPosCur = uvDtBdPixPosCur[uvValidInd.squeeze()].copy()
            trgPatchCur = trgPatchCur[:,:, uvValidInd.squeeze()].copy()
            uvTformCand = uvTformCand[uvValidInd.squeeze(), :].copy()
            uvCostCur = uvCostCur[uvValidInd.squeeze()].copy()
            uvPlaneIDCand = uvPlaneIDCur[uvValidInd.squeeze()].copy()

            uvPixUpdatePos = uvPixActivePos[uvValidInd.squeeze()].copy()

            srcPatch = prep_source_patch(img, uvTformCand, option)

            costPatchCandAll, uvBiasCand = patch_cost(trgPatchCur, srcPatch, modelPlane, uvPlaneIDCand,
                                                      uvPixValid.sub, uvTformCand, srcPosMapCur, uvDtBdPixPosCur,
                                                      option)
            costPatchCand = np.sum(costPatchCandAll, axis=1)

            updateInd = costPatchCand < uvCostCur.squeeze()

            uvPixUpdatePos = uvPixUpdatePos[updateInd.squeeze()]

            numUpdatePix = uvPixUpdatePos.shape[0]

        else:
            numUpdatePix = 0

        if numUpdatePix != 0:
            nUpdateTotal += numUpdatePix

            NNF.uvTform.data[uvPixUpdatePos, :] = uvTformCand[updateInd, :]
            NNF.uvCost.data[uvPixUpdatePos] = costPatchCand[updateInd][..., None]
            NNF.uvPlaneID.data[uvPixUpdatePos] = uvPlaneIDCand[updateInd]

            if option.useBiasCorrection:
                NNF.uvBias.data[:, uvPixUpdatePos] = uvBiasCand[:, updateInd]

            uvPixValidInd = uvPixValid.ind[updateInd]
            NNF.uvTform.map = update_uvMap(NNF.uvTform.map, uvTformCand[updateInd,:], uvPixValidInd)

            NNF.uvCost.map = update_uvMap(NNF.uvCost.map, costPatchCand[updateInd][..., None], uvPixValidInd)

            if len(NNF.uvPlaneID.map.shape) == 2:
                NNF.uvPlaneID.map = NNF.uvPlaneID.map[..., None]
            NNF.uvPlaneID.map = update_uvMap(NNF.uvPlaneID.map,
                                             uvPlaneIDCand[updateInd][..., None], uvPixValidInd)

            uvPixNextSub = uvPixValid.sub[updateInd, :].copy()
            uvPixNextSub = uvPixNextSub + option.propDir[iDirect, :][None, ...]

            updateMap = NNF.uvPix.mask
            updateMap[uvPixNextSub[:, 1], uvPixNextSub[:, 0]] = 0

            uvPixActiveInd = updateMap[NNF.uvPix.sub[:, 1], NNF.uvPix.sub[:, 0]] == 0
            uvPixActiveInd = (uvPixActiveInd == 1) & (uvPixN.validInd == 1)
    return NNF, nUpdateTotal

def random_search(trgPatch, img, NNF, modelPlane, option):
    H, W, Ch = img.shape

    uvPix = NNF.uvPix
    numUvPix = uvPix.sub.shape[0]

    searchRad = max(H, W) / 2
    nUpdateTotal = 0

    iter = 0
    while searchRad > 1:
        iter += 1
        searchRad = searchRad / 2
        if searchRad < 1:
            break

        srcPosMapCur = NNF.uvTform.map[:, :, 6:8]
        uvTformCandCur = uvMat_from_uvMap(NNF.uvTform.map, uvPix.ind)

        srcPos = uvTformCandCur[:, 6:8] + 2 * searchRad * (np.random.rand(numUvPix, 2) - 0.5)

        uvPlaneIDCand = draw_plane_id(NNF.uvPlaneID.planeProbAcc)

        uvTformCand = src_domain_tform(uvPlaneIDCand.squeeze(), modelPlane, [], srcPos, NNF.uvPix.sub, 1)

        uvTformScale = scale_tform(uvTformCand)
        uvValidScaleInd = (uvTformScale.squeeze() > option.minScale) & (uvTformScale.squeeze() < option.maxScale)
        uvValidSrcInd = check_valid_uv(uvTformCand[:, 6:8], NNF.validPix.mask)

        uvValidInd = (uvValidSrcInd == 1).squeeze() & (uvValidScaleInd.squeeze() == 1).squeeze()

        uvPixActivePos = np.array(np.where(uvValidInd.squeeze())).squeeze()

        numActPix = uvPixActivePos.shape[0]

        if numActPix != 0:
            trgPatchCur = trgPatch[:, :, uvValidInd.squeeze()]
            uvCostDataCur = NNF.uvCost.data[uvValidInd]
            uvTformCandCur = uvTformCand[uvValidInd, :].copy()
            uvPlaneIDCandCur = uvPlaneIDCand[uvValidInd].squeeze()

            uvPixValid = type("uvPixValid", (), {})
            uvPixValid.sub = uvPix.sub[uvValidInd, :]
            uvPixValid.ind = uvPix.ind[uvValidInd]

            uvDtBdPixPosCur = NNF.uvDtBdPixPos[uvValidInd].copy()

            srcPatch = prep_source_patch(img, uvTformCandCur, option)

            [costPatchCandAll, uvBiasCand] = patch_cost(trgPatchCur, srcPatch, modelPlane, uvPlaneIDCandCur,
                                    uvPixValid.sub, uvTformCandCur, srcPosMapCur, uvDtBdPixPosCur, option)
            costPatchCand = np.sum(costPatchCandAll, axis=1)
            updateInd = (costPatchCand.squeeze() < uvCostDataCur.squeeze())
            nUpdate = np.sum(updateInd)

            if nUpdate != 0:

                uvPixActivePos = uvPixActivePos[updateInd]

                nUpdateTotal = nUpdateTotal + nUpdate

                NNF.uvTform.data[uvPixActivePos, :] = uvTformCandCur[updateInd, :].copy()
                NNF.uvPlaneID.data[uvPixActivePos] = uvPlaneIDCandCur[updateInd].copy()

                NNF.uvCost.data[uvPixActivePos] = costPatchCand[updateInd][..., None].copy()
                if option.useBiasCorrection:
                    NNF.uvBias.data[:, uvPixActivePos] = uvBiasCand[:, updateInd]

                uvPixValidInd = uvPixValid.ind[updateInd].copy()
                NNF.uvTform.map = update_uvMap(NNF.uvTform.map, uvTformCandCur[updateInd,:], uvPixValidInd)

                NNF.uvPlaneID.map = update_uvMap(NNF.uvPlaneID.map, uvPlaneIDCandCur[updateInd][..., None],
                                                 uvPixValidInd)
                NNF.uvCost.map = update_uvMap(NNF.uvCost.map, costPatchCand[updateInd][..., None],
                                              uvPixValidInd)

    return NNF, nUpdateTotal

def update_NNF(trgPatch, img, NNF, modelPlane, modelReg, option):
    nUpdate = np.zeros((3))

    for i in range(option.numPassPerIter):
        for iDierct in range(4):
            NNF, n = propagate(trgPatch, img, NNF, modelPlane, option, iDierct)
            nUpdate[0] += n

        NNF, n = random_search(trgPatch, img, NNF, modelPlane, option)
        nUpdate[1] += n
    return NNF, nUpdate