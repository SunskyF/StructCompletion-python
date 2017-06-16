import cv2
import numpy as np
import os
import math
from sklearn.cluster import MeanShift

def extract_regularity(img, maskD, modelPlane, option):
    '''
    
    :param img: 
    :param maskD: 
    :param modelPlane: 
    :param option: 
    :return: 
    '''
    img = img.astype(np.float32)
    imgGray = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)

    H, W = imgGray.shape

    option.PeakThreshold = 0.12
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=option.PeakThreshold)

    kp, des = sift.detectAndCompute(imgGray, None)

    if len(kp) > option.maxNumLocalFeat:
        index = np.random.permutation(len(kp))[:option.maxNumLocalFeat]
        kp_temp = []
        des_temp = []
        for ind in index:
            kp_temp.append(kp[ind])
            des_temp.append(des[ind])
        kp = kp_temp
        des = des_temp

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des, des, k=option.numQueryNN)

    index = np.zeros((option.numQueryNN, len(kp)), dtype=np.int)
    distance = np.zeros((option.numQueryNN, len(kp)))
    keypoints = np.zeros((len(kp), 2))
    for i in range(len(kp)):
        keypoints[i] = [kp[i].pt[0], kp[i].pt[1]]
        for j in range(option.numQueryNN):
            index[j, i] = matches[i][j].trainIdx
            distance[j, i] = matches[i][j].distance

    featMatchData = type("featMatchData", (), {})
    featMatchData.index = index
    featMatchData.distance = distance
    featMatchData.K = option.numQueryNN
    featMatchData.frames = keypoints.T
    featMatchData.numFeat = len(kp)

    for indPlane in range(modelPlane.numPlane):
        H = np.eye(3)
        H[2, :] = modelPlane.plane[indPlane].vline

        framesRect = np.vstack([featMatchData.frames[:2, :], np.ones((1, featMatchData.numFeat))])
        framesRect = H.dot(framesRect)
        framesRect = framesRect / framesRect[2, :]

        nn = featMatchData.index[:featMatchData.K+1, :]

        targetPosRect = np.hstack([framesRect[:2, nn[0, :]], framesRect[:2, nn[0, :]], framesRect[:2, nn[1, :]]])
        sourcePosRect = np.hstack([framesRect[:2, nn[1, :]], framesRect[:2, nn[2, :]], framesRect[:2, nn[2, :]]])

        targetPos = np.hstack([featMatchData.frames[:2, nn[0, :]],
                               featMatchData.frames[:2, nn[0, :]],
                               featMatchData.frames[:2, nn[1, :]]])
        sourcePos = np.hstack([featMatchData.frames[:2, nn[1, :]],
                               featMatchData.frames[:2, nn[2, :]],
                               featMatchData.frames[:2, nn[2, :]]])

        targetPos = np.round(targetPos).astype(np.int)
        sourcePos = np.round(sourcePos).astype(np.int)

        # compute matching weights

        planeProb = modelPlane.postProb[:, :, indPlane]
        weightVec = planeProb[targetPos[1, :], targetPos[0, :]] * planeProb[sourcePos[1, :], sourcePos[0, :]]

        # get the matched features on the plane
        validMatchInd = weightVec >= option.prodProbThres
        sourcePosRect = sourcePosRect[:, validMatchInd]
        targetPosRect = targetPosRect[:, validMatchInd]

        # compute the displacement vectors in the rectified space
        dispVecRect = sourcePosRect - targetPosRect
        distDispVecRect = np.sum(dispVecRect ** 2, axis=0)

        validDispVecInd = distDispVecRect > option.minDistDispVec
        dispVecRect = dispVecRect[:, validDispVecInd]

        dispVecRect = np.hstack([dispVecRect, -dispVecRect])
        dispVecRect = np.hstack([dispVecRect, 2 * dispVecRect])

        if dispVecRect.shape[1] != 0:
            clf = MeanShift(bandwidth=option.msBandwidth)
            clf.fit(dispVecRect.T)

            clustCent = clf.cluster_centers_

            numMemberInCluster = np.zeros((clustCent.shape[0]))
            for clusterInd in range(numMemberInCluster.shape[0]):
                numMemberInCluster[clusterInd] = (clf.labels_ == clusterInd).sum()

            validClusterInd = numMemberInCluster >= option.msMinNumClusterMember

            clustCent = clustCent[validClusterInd, :]
            modelPlane.plane[indPlane].dispVec = clustCent.T # (numDim, numClust)
            modelPlane.plane[indPlane].numDispVec = clustCent.shape[0]

        else:
            modelPlane.plane[indPlane].dispVec = np.array([])
            modelPlane.plane[indPlane].numDispVec = 0

    return modelPlane
