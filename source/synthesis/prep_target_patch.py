import numpy as np

def prep_target_patch(img, uvPixSub, option):
    imgH, imgW, Ch = img.shape

    numUvPix = uvPixSub.shape[1]

    uvPixSub = np.reshape(uvPixSub, (1, 2, numUvPix))

    refPatPos = option.refPatchPos[:, :2][..., None]
    trgPatchPos = (refPatPos + uvPixSub).astype(np.int)

    trgPatch = np.zeros((option.pNumPix, Ch, numUvPix))
    for i in range(numUvPix):
        trgPatch[:, :, i] = img[trgPatchPos[:, 1, i], trgPatchPos[:, 0, i], :].reshape(
            (option.pSize, option.pSize, Ch)).reshape((option.pNumPix, Ch), order="F")  # 需要reshape成9 * 9才是正确的

    return trgPatch