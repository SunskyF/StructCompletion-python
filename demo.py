import time
import cv2
import os

from init_opt import optA, optS
from source.extract.extract_planar_structure import extract_planar_structure
from source.pyramid.create_pyramid import create_pyramid
from source.pyramid.planar_structure_pyramid import planar_structure_pyramid
from source.synthesis.synthesis import synthesis
from source.poisson_blend import poisson_blend

if __name__ == "__main__":
    image_id = 1
    image_name = "%03d_input_hole.png" % image_id

    print('- Extract planar structures')
    time0 = time.time()
    img, mask, maskD, modelPlane, modelReg = extract_planar_structure(image_name, optA)
    print('Done in %6.3f seconds.\n' % (time.time() - time0))

    print('- Construct image pyramid')
    time1 = time.time()
    imgPyr, maskPyr, scaleImgPyr = create_pyramid(img, maskD, optS)

    modelPlane, modelReg = planar_structure_pyramid(scaleImgPyr, modelPlane, modelReg)
    print('Done in %6.3f seconds.\n' % (time.time() - time1))

    # Completion by synthesis
    time2 = time.time()
    print('- Image completion using planar structure guidance')
    imgPyr = synthesis(imgPyr, maskPyr, modelPlane, modelReg, optS)
    print('Synthesis took %6.3f seconds.\n' % (time.time() - time2))

    imgSyn = imgPyr[0]
    cv2.imshow("Result", imgSyn)
    cv2.waitKey()
    # Poisson blending
    time3 = time.time()
    imgCompleteFinal = poisson_blend(img, imgSyn, mask)
    print('Blending took %6.3f seconds.\n' % (time.time() - time3))
    if not os.path.exists("result"):
        os.makedirs("result")

    cv2.imwrite(os.path.join("result", str(image_id) + "_completion.png"), imgCompleteFinal)