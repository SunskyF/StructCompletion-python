import numpy as np

def draw_plane_id(planeProbAccData):
    numUvPix = planeProbAccData.shape[0]
    numPlane = planeProbAccData.shape[1] - 1

    randSample = np.random.rand(numUvPix)
    uvPlaneIDData = np.zeros((numUvPix, 1), dtype=np.uint8)

    for indPlane in range(numPlane):

        indSamplePlane = (planeProbAccData[:, indPlane] < randSample) & \
                         (planeProbAccData[:, indPlane+1] >= randSample)

        uvPlaneIDData[indSamplePlane == 1] = indPlane

    return uvPlaneIDData
