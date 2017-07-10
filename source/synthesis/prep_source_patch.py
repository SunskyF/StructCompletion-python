import numpy as np
import cv2
from scipy.interpolate import interp2d

def prep_source_patch(img, uvTform, option):
    numUvPix = uvTform.shape[0]

    c1 = np.reshape(uvTform[:, :3].T, (1, 3, numUvPix), order="F")
    c2 = np.reshape(uvTform[:, 3:6].T, (1, 3, numUvPix), order="F")
    c3 = np.reshape(uvTform[:, 6:9].T, (1, 3, numUvPix), order="F")

    srcPatchPos = (option.refPatchPos[:, 0][..., None, None] * c1) + \
                  (option.refPatchPos[:, 1][..., None, None] * c2)
    srcPatchPos += c3

    srcPatchPos /= srcPatchPos[:, 2:3, :]

    srcPatch = cv2.remap(img, srcPatchPos[:, 0, :].astype(np.float32), srcPatchPos[:, 1, :].astype(np.float32),
                         cv2.INTER_LINEAR)

    srcPatch = srcPatch.transpose((0, 2, 1))

    return srcPatch