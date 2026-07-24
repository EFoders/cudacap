// COMMON /DATR/ — the Gaussian-integration setup GETHP uses; the CUDA C twin
// of scalar/datr.py.
//
// XT/WT are the 20-point half Gauss-Legendre abscissae/weights from
// blkdat.for (GETHP evaluates the folded 40-point rule via the 1-XT / 1+XT
// symmetric pair).  NT/NPL/XNPL/TWDIV are set once in voacapw.for's startup:
// a 20-point rule with the linear (NPL=1) interval transformation, so
// TWDIV = 1/2**NPL = 0.5 and XNPL = 1.0.  A C float literal rounds its
// decimal to nearest-even binary32 exactly as np.float32 does, so the
// tables carry the same bit patterns as Stage A's.

#ifndef VOACAP_DATR_CUH
#define VOACAP_DATR_CUH

#define DATR_NT    20
#define DATR_NPL   1
#define DATR_XNPL  1.0f
#define DATR_TWDIV 0.5f            // 1 / 2**NPL

// 1-based (21,): XT[1..20], WT[1..20]; slot 0 padded.
__device__ __constant__ float DATR_XT[21] = {
    0.0f,
    0.0387724175f, 0.1160840707f, 0.1926975807f, 0.2681521850f,
    0.3419940908f, 0.4137792043f, 0.4830758017f, 0.5494671251f,
    0.6125538897f, 0.6719566846f, 0.7273182552f, 0.7783056514f,
    0.8246122308f, 0.8659595032f, 0.9020988070f, 0.9328128083f,
    0.9579168192f, 0.9772599500f, 0.9907262387f, 0.9982377097f,
};

__device__ __constant__ float DATR_WT[21] = {
    0.0f,
    0.0775059480f, 0.0770398182f, 0.0761103619f, 0.0747231691f,
    0.0728865824f, 0.0706116474f, 0.0679120458f, 0.0648040135f,
    0.0613062425f, 0.0574397691f, 0.0532278470f, 0.0486958076f,
    0.0438709082f, 0.0387821680f, 0.0334601953f, 0.0279370070f,
    0.0222458492f, 0.0164210584f, 0.0104982845f, 0.0045212771f,
};

#endif  // VOACAP_DATR_CUH
