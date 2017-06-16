import cv2
import numpy as np
import os
import math

def planar_structure_pyramid(scaleImgPyr, modelPlane, modelReg):
    numLevel = len(scaleImgPyr)

    modelPlanePyr = []
    modelRegPyr = []

    planePostProb = modelPlane.postProb

    for ilvl in range(numLevel):
        scaleImgCur = scaleImgPyr[ilvl][0]

        sizeImgCur = scaleImgPyr[ilvl][1]

        planePostProbCur = cv2.resize(planePostProb, (sizeImgCur[1], sizeImgCur[0]), interpolation=cv2.INTER_CUBIC)
        planePostProbCursum = np.sum(planePostProbCur, axis=2)
        planePostProbCur = planePostProbCur / planePostProbCursum[..., None]

        singlePlanePyr = type('PlanePyr', (), {})
        singlePlanePyr.numPlane = modelPlane.numPlane
        singlePlanePyr.planeProb = planePostProbCur
        singlePlanePyr.mLogPlaneProb = -np.log(planePostProbCur)
        singlePlanePyr.rectMat = []
        singlePlanePyr.rotPar = []
        singlePlanePyr.rotMat = []
        modelPlanePyr.append(singlePlanePyr) # numPlane, planePostProbCur, -log(planePostProbCur)

        for iplane in range(modelPlane.numPlane):
            H = np.eye(3)
            vline = modelPlane.plane[iplane].vline.copy()

            vline[:1] = vline[:1] / scaleImgCur
            H[2, :] = vline

            modelPlanePyr[ilvl].rectMat.append(H)
            modelPlanePyr[ilvl].rotPar.append(modelPlane.plane[iplane].rotPar)
            modelPlanePyr[ilvl].rotMat.append([])

            for itheta in range(2):
                t = modelPlanePyr[ilvl].rotPar[iplane][itheta]
                Hr = np.eye(3)
                Hr[0, 0] = np.cos(t)
                Hr[0, 1] = -np.sin(t)
                Hr[1, 0] = np.sin(t)
                Hr[1, 1] = np.cos(t)

                modelPlanePyr[ilvl].rotMat[iplane].append(Hr)

        singleRegPyr = type('RegPyr', (), {})
        dispVec = []
        numDispVec = []

        for iplane in range(modelPlane.numPlane):
            dispVec.append(scaleImgCur * modelReg.plane[iplane].dispVec)
            numDispVec.append(modelReg.plane[iplane].numDispVec)

        singleRegPyr.dispVec = dispVec
        singleRegPyr.numDispVec = numDispVec
        modelRegPyr.append(singleRegPyr)

    return modelPlanePyr, modelRegPyr