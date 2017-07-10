import cv2
import numpy as np
import os
import math
from scipy import ndimage

class vp():
    def __init__(self, pos, score, numLines):
        self.pos = pos
        self.score = score
        self.numLines = numLines
        self.lines = None

class VP_imgLines():
    def __init__(self, imgLines, imgLinesPosMap):
        self.imgLines = imgLines
        self.imgLinesPosMap = imgLinesPosMap

class Plane():
    def __init__(self, vline, imgPlaneProb, sourceVP):
        self.vline = vline

        self.imgPlaneProb = imgPlaneProb
        self.score = imgPlaneProb.sum()
        self.sourceVP = sourceVP
        self.rotPar = [0, 0]

        self.dispVec = None
        self.numDispVec = None

def vLineFromTwoVP(vp1, vp2):
    A = np.vstack([vp1, vp2])
    u, s, v = np.linalg.svd(A)

    vLine = v.T[:, -1]
    vLine /= vLine[2]
    return vLine

def gaussian_kernel(ksize, sigma):
    kernel = np.zeros((ksize[0], ksize[1]), dtype=np.float)
    halfsizeX = ksize[0] // 2
    halfsizeY = ksize[1] // 2

    sigma2 = 2 * sigma * sigma
    n_halfsizeX = halfsizeX
    n_halfsizeY = halfsizeY

    half_one_x = 0
    half_one_y = 0
    if ksize[0] % 2 == 0:
        n_halfsizeX -= 1
        half_one_x = 0.5
    if ksize[1] % 2 == 0:
        n_halfsizeY -= 1
        half_one_y = 0.5
    for i in range(n_halfsizeX + 1):
        for j in range(n_halfsizeY + 1):
            value = math.exp(-((i + half_one_x) ** 2 + (j + half_one_y) ** 2) / sigma2)
            kernel[halfsizeX + i][halfsizeY + j] = value
            kernel[halfsizeX + i][n_halfsizeY - j] = value
            kernel[n_halfsizeX - i][halfsizeY + j] = value
            kernel[n_halfsizeX - i][n_halfsizeY - j] = value
    kernel = kernel / kernel.sum()

    return kernel

def detect_plane_from_vp(vpData, img, mask, option):
    '''

    :param vpData: 
    :param img: 
    :param mask: 
    :param option: 
    :return: 
    '''
    assert (img.shape[-1] == 3)
    height, width, channel = img.shape
    HfilterX = gaussian_kernel([1, option.filterSize], option.filterSigma)
    HfilterY = HfilterX.T

    modelPlane = type("modelPlane", (), {"vp": []})

    #  first estimate the spatial support of each VP by diffusing
    # its corresponding line segments using a wide Gaussian kernel
    for i in range(vpData.numVP):
        imgLines = np.zeros((height, width))
        for line in vpData.vp[i].lines:
            cv2.line(imgLines, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, 1)

        #cv2.imshow("tmp", imgLines)
        #cv2.waitKey()
        imgLines = imgLines.astype(np.double) / 255
        imgLinesPosMap = imgLines.copy()

        for k in range(option.numFilterIter):
            imgLinesPosMap = cv2.filter2D(imgLinesPosMap, -1, HfilterX, borderType=cv2.BORDER_REPLICATE)

        for k in range(option.numFilterIter):
            imgLinesPosMap = cv2.filter2D(imgLinesPosMap, -1, HfilterY, borderType=cv2.BORDER_REPLICATE)

        # Save results
        modelPlane.vp.append(VP_imgLines(imgLines, imgLinesPosMap))

    # Estimate plane support and plane parameters
    numPlane = vpData.numVP * (vpData.numVP - 1) // 2

    # Initialize plane data
    modelPlane.plane = []

    # A pair of vanishing points forms a plane hypothesis
    for i in range(vpData.numVP - 1):
        for j in range(i + 1, vpData.numVP):
            modelPlane.plane.append(Plane(vLineFromTwoVP(vpData.vp[i].pos, vpData.vp[j].pos),
                                          modelPlane.vp[i].imgLinesPosMap * modelPlane.vp[j].imgLinesPosMap,
                                          [i, j]))


    for i in range(numPlane):
        for vpInd in [0, 1]:
            linesCurr = np.array(vpData.vp[modelPlane.plane[i].sourceVP[vpInd]].lines)

            invalidLieInd = linesCurr[:, 4] == 0 # 长度不是0的线段

            linesCurr = linesCurr[invalidLieInd == 0, :]
            numLines = linesCurr.shape[0]

            vLineCurr = modelPlane.plane[i].vline

            # Rectified homography
            H = np.eye(3) # 平移变换
            H[2, :] = vLineCurr


            linesStart = np.hstack([linesCurr[:, :2], np.ones((numLines, 1))]).T
            linesEnd = np.hstack([linesCurr[:, 2:4], np.ones((numLines, 1))]).T

            linesStartRect = H.dot(linesStart)
            linesStartRect = linesStartRect / np.vstack([linesStartRect[2, :],
                                                         linesStartRect[2, :],
                                                         linesStartRect[2, :]])

            linesEndRect = H.dot(linesEnd)
            linesEndRect = linesEndRect / np.vstack([linesEndRect[2, :],
                                                     linesEndRect[2, :],
                                                     linesEndRect[2, :]])

            linesVec = linesStartRect[:2, :] - linesEndRect[:2, :]
            linesSign = linesEndRect[1, :] > linesStartRect[1, :]
            linesSign = 2 * linesSign - 1

            linesLength = np.sqrt(np.sum(linesVec ** 2, axis=0))
            linesCos = linesSign * linesVec[0, :] / linesLength

            theta = np.arccos(linesCos)
            thetaAvg = np.mean(theta)

            for iter in range(5):
                thetadiff = theta - thetaAvg
                indLargeTheat = thetadiff > math.pi / 2
                theta[indLargeTheat] = math.pi - theta[indLargeTheat]

                indSmallTheta = thetadiff < -math.pi / 2
                theta[indSmallTheta] = math.pi + theta[indSmallTheta]
                thetaAvg = np.mean(theta)

            modelPlane.plane[i].rotPar[vpInd] = thetaAvg

    # add ad fronto-parallel plane
    modelPlane.plane.append(Plane(np.array([0, 0, 1]), option.fpPlaneProb*np.ones((height, width)), 0))
    numPlane += 1
    modelPlane.numPlane = numPlane

    # compute posterior prob
    planeProb = np.zeros((height, width, numPlane))
    for i in range(numPlane):
        planeProb[:, :, i] = modelPlane.plane[i].imgPlaneProb

    planeProbSum = np.sum(planeProb, axis=2)
    planeProb = planeProb / planeProbSum[..., None]
    modelPlane.postProbHole = planeProb

    edt, inds = ndimage.distance_transform_edt(1-(mask == 0), return_indices=True)

    maskInt = mask.copy()
    maskInt[0, :] = 0
    maskInt[-1, :] = 0
    maskInt[:, 0] = 0
    maskInt[:, -1] = 0

    # propagate posterior prob into the hole region
    for i in range(numPlane):
        planeProbCh = planeProb[:, :, i]
        planeProb[:, :, i] = planeProbCh[inds[0, :, :], inds[1, :, :]].copy()
        #cv2.imshow("postProb", planeProb[:, :, i])
        #cv2.waitKey()


    planeProbSum = np.sum(planeProb, axis=2)
    planeProb = planeProb / planeProbSum[..., None]

    planeProbSum = 1 + numPlane*option.probConst
    planeProb = (planeProb + option.probConst) / planeProbSum

    modelPlane.postProb = planeProb.copy()
    #print(modelPlane.postProb.shape)
    #cv2.imshow("postProb", modelPlane.postProb)
    #cv2.waitKey()
    return modelPlane

def read_vpdata(fileName):
    '''

    :param fileName: vp filename
    :return: VP: VPdata
    '''
    VP = type("VPData", (), {"numVP": 0, "vp": []})
    with open(fileName, 'r') as f:
        f.readline()
        while True:
            temp = f.readline()
            if temp == "\n":
                f.readline()
                break
            numbers = temp.split()
            record = list(map(float, numbers))
            VP.vp.append(vp(np.array(record[:3]), record[3], int(record[4])))
            VP.numVP += 1

        allLines = f.readlines()
        nowLine = 0
        for i in range(VP.numVP):
            numLines = VP.vp[i].numLines
            assert (int(allLines[nowLine]) == numLines)

            def clean(line):
                numbers = line.split()
                record = list(map(float, numbers))
                return record

            VP.vp[i].lines = list(map(clean, allLines[nowLine + 1:nowLine + numLines + 1]))
            assert (len(VP.vp[i].lines) == numLines)
            nowLine += numLines + 1

    return VP

def extract_plane(image_name, img, mask, option):
    '''
        extract plane model from an image

    :param image_name: 
    :param img: 
    :param maskD: 
    :param option: 
    :return: modelPlane
    '''
    # VP detection
    vpFilePath = 'cache/vpdetection'
    vpFileName = image_name[:-4] + '-vanishingpoints.txt'

    if not os.path.exists(os.path.join(vpFilePath, 'text', vpFileName)):
        vpDetectCMD = 'vpdetection.exe -indir data -infile ' + image_name + ' -outdir ' + vpFilePath
        print("Using CMD: ", vpDetectCMD)
        os.system(vpDetectCMD)

    # 获得三个消失点和对应的线段
    vpData = read_vpdata(os.path.join(vpFilePath, 'text', vpFileName))

    modelPlane = detect_plane_from_vp(vpData, img, mask, option)

    return modelPlane