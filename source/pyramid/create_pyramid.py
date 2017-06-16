import cv2
import numpy as np
import os
import math
from scipy import ndimage

def init_coarsest_level(img, mask):

    _, idMap = ndimage.distance_transform_edt(1 - (mask == 0), return_indices=True)
    maskInt = mask.copy().astype(np.uint8)
    maskInt[0, :] = 0
    maskInt[-1, :] = 0
    maskInt[:, 0] = 0
    maskInt[:, -1] = 0

    for ch in range(3):
        imgCh = img[:, :, ch]
        imgCh = imgCh[idMap[0, :, :], idMap[1, :, :]]
        img[:, :, ch] = imgCh
    #cv2.imshow("tmp", img)
    #cv2.waitKey()
    # TODO: roifill
    #temp = cv2.blur(img, (3, 3))

    #img[maskInt == 1] = temp[maskInt == 1]
    #img = cv2.inpaint((img* 255).astype(np.uint8), maskInt, 3, cv2.INPAINT_TELEA)

    return img

def create_scale_pyramid(h, w, option):
    min_size = min(h, w)
    coarestScale = option.coarestImgSize / min_size

    scalePyr = None
    if option.useLogScale:
        scalePyr = 2 ** np.linspace(0, np.log2(coarestScale), option.numPyrLvl)

    imgHpyr = np.round(h * scalePyr)
    imgWpyr = np.round(w * scalePyr)

    scaleImgPyr = []
    scaleImgPyr.append([1, [h, w]]) # imgscale, imgsize[h, w]
    for k in range(1, option.numPyrLvl):
        scaleImgPyr.append([scalePyr[k], [int(imgHpyr[k]), int(imgWpyr[k])]])

    return scaleImgPyr

def create_image_pyramid(img, scaleImgPyr, imageType, option):
    imgPyr = [img]
    for ilvl in range(1, option.numPyrLvl):
        imgHcurlvl = scaleImgPyr[ilvl][1][0]
        imgWcurlvl = scaleImgPyr[ilvl][1][1]

        imgCur = imgPyr[ilvl - 1]

        imgPyr.append(cv2.resize(imgCur, (imgWcurlvl, imgHcurlvl), interpolation=cv2.INTER_CUBIC))
        #cv2.imshow("tmp", imgPyr[ilvl])
        #cv2.waitKey()

    if imageType == "mask":
        for ilvl in range(1, option.numPyrLvl):
            imgPyr[ilvl] = (imgPyr[ilvl] > 0.5).astype(np.float32)

    return imgPyr

def create_pyramid(img, mask, option):
    img_copy = img.copy()
    H, W, ch = img.shape

    img = init_coarsest_level(img_copy, mask)

    scaleImgPyr = create_scale_pyramid(H, W, option)

    maskPyr = create_image_pyramid(mask, scaleImgPyr, 'mask', option)
    imgPyr = create_image_pyramid(img, scaleImgPyr, 'image', option)

    imgPyr[-1] = init_coarsest_level(imgPyr[-1], maskPyr[-1])

    return imgPyr, maskPyr, scaleImgPyr