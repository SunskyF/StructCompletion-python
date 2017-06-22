import math
import numpy as np
print('- Initialize parameters')

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

optA = type('opt_A', (), {})
optS = type('opt_A', (), {})

#  Synthesis: guided completion
# =========================================================================
# Patch size
# =========================================================================
optS.pSize = 9                        # Patch size (odd number), use larger patch for more coherent region
optS.pRad  = math.floor(optS.pSize / 2)      # Patch radius
optS.pNumPix = optS.pSize * optS.pSize  # Number of pixels in a patch
optS.pMidPix = round(optS.pNumPix / 2)  # The center of the patch

# =========================================================================
# Multi-resolution parameters
# =========================================================================
optS.numPyrLvl = 10                   # Number of coarse to fine layer
optS.coarestImgSize = 32              # The size of the smallest image in the pyramid
optS.useLogScale = 1                  # Use log scales or linear scales for downsampling

# Weighting parameters for patch match, larger weight put more emphasis on
# pixels near to known pixels
optS.wDist = 2 ** np.linspace(1, 0.0, optS.numPyrLvl)

# Patch weighting kernel
h = gaussian_kernel([optS.pSize, optS.pSize], optS.pRad)
h = h.reshape(-1) / np.sum(h)
optS.wPatch = h

# =========================================================================
# Parameters for domain transformation and photometric compensation
# =========================================================================
# This scale range is used to reject unlikely patch transformation
optS.minScale = 0.75                   # Mininum patch scale variation
optS.maxScale = 1.25                   # Maximum patch scale variation

# Parameters for photometric compensation
optS.minBias = -0.05                   # Mininum bias compensation
optS.maxBias =  0.05                   # Maximum bias compensation

# =========================================================================
# Coarse-to-fine iterations
# =========================================================================
# Number of iterations per level
optS.numIter    = 10                  # The initial iteration number, large hole might require
                                       # more iterations
optS.numIterDec = optS.numIter / optS.numPyrLvl   # Number of decrements
optS.numIterMin = 3                   # Minimum number of iterations
optS.numPassPerIter = 1

# =========================================================================
# Weighting parameters
# =========================================================================
# To-Do: adaptive parameter selection
optS.lambdaCoherence = 1e-2            # Weighting parameter for coherence

# Planar cost
optS.lambdaPlane  = 5e-2               # Weighting parameters for planar cost

# Directional cost
optS.lambdaDirect = 5                  # Weighting parameters for directional cost
optS.directThres  = 0.05               # Directional cost threshold

optS.lambdaProx   = 0e-2               # Weighting parameters for proximity cost
optS.proxThres    = 0.25

optS.lambdaReg    = -0.02              # Weighting parameters for encouraging
# regularity-guided sampling

# =========================================================================
# Method configuration
# =========================================================================

optS.useRegGuide   = 1
optS.usePlaneGuide = 1
optS.useBiasCorrection = 1

# === Precomputed patch position in the reference position ===
X, Y = np.meshgrid(np.arange(-optS.pRad, optS.pRad+1), np.arange(-optS.pRad, optS.pRad+1))
optS.refPatchPos = np.vstack([Y.reshape(-1), X.reshape(-1), np.ones(optS.pSize*optS.pSize)]).T

# === Propagation directions ===
optS.propDir = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

optS.rsThres =  0.05                  # Random sampling threshold for early termination
optS.voteUpdateW = 1                    # Smaller number for quick voting update

optS.numRegSample  = 5                # Number of regularity-guided sampling per iteration
optS.numRandSample = 5                # Number of coarse-to-fine random sampling per iteration

# [To-Do] robust norm, e.g., huber
optS.costType = 'L1'                  # Patch appearance cost, other option: 'L2'

# Analysis: extracting planar structure
# === Plane parameters ===
# Detect up to 3 vanishing points, always include dummy VP (froto-parallel plane)
optA.numVP = 4

# Fixed density for frontal parallel plane [To-Do] should adapt to image size
optA.fpPlaneProb = 1e-4

# === Regularity parameters ===
# Maximum number of local features
optA.numFeatMatch = 2000

# Blurring operations for estimating the plane spatial support
optA.filterSize = 100
optA.filterSigma = 50
optA.numFilterIter = 20

# Add the constant to all plane posterior probabilities
optA.probConst = 0.05

# Parameters for feature extraction and matching
optA.PeakThreshold = 0.04
optA.maxNumLocalFeat = 1000
optA.numQueryNN = 3
optA.maxDispVec = 1000
optA.minDistDispVec = 100

# Parameters for the Mean-Shift clustering algorithm
optA.msBandwidth = 20
optA.msMinNumClusterMember = 5

# Threshold for determining whether a pair of feature matches is on a plane
optA.prodProbThres = 0.5