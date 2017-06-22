import cv2
import numpy as np
import os
import math

from .extract_regularity import extract_regularity
from .extract_plane import extract_plane

def extract_planar_structure(image_name, option):
    '''
    :param image_name: 
    :param option: 
    :return: 
        img:        Original image
        mask:       Hole mask
        modelPlane: Plane model
        modelReg:   Regularity model
    '''
    img = cv2.imread(os.path.join('data', image_name), cv2.IMREAD_UNCHANGED)
    #cv2.imshow("origin", img)
    #cv2.waitKey()
    assert(img.shape[-1] == 4)

    alpha = img[..., -1]
    mask = (alpha != 255).astype(np.uint8)

    image = img[..., :3].astype(np.double).copy() / 255

    mask = cv2.dilate(mask, np.ones((5, 5)))

    maskD = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

    # Extract planar constructural constraints
    modelPlane = extract_plane(image_name, image, maskD, option)

    # Extract regularity constraints
    modelReg = extract_regularity(image, maskD, modelPlane, option)

    return image, mask.astype(np.double), maskD.astype(np.double), modelPlane, modelReg


