// GEOM — circuit path geometry: great-circle distance and bearings (with the
// NPSL=1 long-path/through-antipode option), the 1/3/5 ionospheric sample
// areas along the path, their geographic and geomagnetic coordinates,
// magnetic dip/gyrofrequency (via MAGVAR) and per-area ground constants.
// The CUDA C twin of scalar/geom.py.
//
// The land/sea decision is NOISY mode 7 (the land-mass map).  Stage A injects
// it as a ``wld_fn`` seam because its oracle dump driver stubs the map; on the
// device the real map is already resident for the rest of the kernel, so the
// call is made directly and the five REDMAP overlay pointers pass through.
//
// All REAL*4; transcendentals via f32math.  Fortran statement labels are
// cited so this audits line-against-line with geom.for / geom.py.

#ifndef VOACAP_GEOM_CUH
#define VOACAP_GEOM_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "magfin.cuh"
#include "noisy.cuh"

#define VC_GEOM_EPSLON 1.0e-7f
#define VC_GEOM_DDX    0.03f    // closest Tx & Rx can be (degrees)

// The /DON/, /RON/ and /GEOG/ fields GEOM writes.  Arrays are length 5;
// entries beyond KM stay zero, as in Stage A's dump driver.
struct GeomResult {
    float dlong, gcd, gcdkm;
    float btr, btrd, brtd;
    float alatd, rlatd;         // GEOM may nudge the input rlatd
    int km;
    float rd[5], clat[5], clong[5], glat[5], gyz[5], gmdip[5];
    float sigpat[5], epspat[5];
};

__device__ inline float geom_clamp1_(float qcos) {
    if (fabsf(qcos) > 1.0f)
        qcos = copysignf(1.0f, qcos);
    return qcos;
}

// Angles in degrees (south/west negative); pole from north_pole.txt.
__device__ inline void geom(float tlatd, float tlongd, float rlatd,
                            float rlongd, int npsl, float pole_lat,
                            float pole_lon, const float* fakp,
                            const float* fakmap, const float* hmym,
                            const float* fakabp, const float* abmap,
                            GeomResult* out) {
    for (int i = 0; i < 5; ++i) {
        out->rd[i] = 0.0f;
        out->clat[i] = 0.0f;
        out->clong[i] = 0.0f;
        out->glat[i] = 0.0f;
        out->gyz[i] = 0.0f;
        out->gmdip[i] = 0.0f;
        out->sigpat[i] = 0.0f;
        out->epspat[i] = 0.0f;
    }

    float glt = pole_lat * VC_D2R;          // geom.for:40-41
    float glg = pole_lon * VC_D2R;
    // 42-47: coincident endpoints — move Rx away from Tx a little bit
    if (fabsf(tlatd - rlatd) <= VC_GEOM_DDX &&
        fabsf(tlongd - rlongd) <= VC_GEOM_DDX) {
        float dd = VC_GEOM_DDX;
        if (rlatd < 0.0f)
            dd = -VC_GEOM_DDX;
        rlatd = tlatd - dd;
    }
    out->rlatd = rlatd;
    float tlat = tlatd * VC_D2R;            // 48-51
    float tlong = tlongd * VC_D2R;
    float rlat = rlatd * VC_D2R;
    float rlong = rlongd * VC_D2R;
    if (fabsf(rlatd) > 89.9f)               // 52-53: at poles, force long=0
        rlong = 0.0f;
    if (fabsf(tlatd) > 89.9f)
        tlong = 0.0f;

    // 55-66: great circle distance
    float dlong = tlong - rlong;            // label 100
    if (fabsf(dlong) > VC_PI)
        dlong = dlong - copysignf(VC_PI2, dlong);
    if (npsl == 1)                          // long great circle path
        dlong = dlong - copysignf(VC_PI2, dlong);
    float qcos = sin32(tlat) * sin32(rlat)
               + cos32(tlat) * cos32(rlat) * cos32(dlong);
    qcos = geom_clamp1_(qcos);
    float gcd = acos32(qcos);
    if (gcd < 0.000001f)                    // minimum distance is 31.85 metres
        gcd = 0.000001f;
    if (npsl == 1)
        gcd = VC_PI2 - gcd;
    float gcdkm = gcd * VC_RZ;
    out->dlong = dlong;
    out->gcd = gcd;
    out->gcdkm = gcdkm;

    // 68-82: bearing transmitter -> receiver
    float btr;
    if (cos32(tlat) - VC_GEOM_EPSLON <= 0.0f) {     // arithmetic IF 101,101,104
        btr = (tlat <= 0.0f) ? 0.0f : VC_PI;        // 102/103
    } else {                                        // 104
        qcos = (sin32(rlat) - sin32(tlat) * cos32(gcd))
             / (cos32(tlat) * sin32(gcd));
        qcos = geom_clamp1_(qcos);
        btr = acos32(qcos);
    }
    if (dlong > 0.0f)                               // 82
        btr = VC_PI2 - btr;
    // 84-98: bearing receiver -> transmitter
    float brt;
    if (cos32(rlat) - VC_GEOM_EPSLON <= 0.0f) {     // 111,111,114
        brt = (rlat <= 0.0f) ? 0.0f : VC_PI;        // 112/113
    } else {                                        // 114
        qcos = (sin32(tlat) - sin32(rlat) * cos32(gcd))
             / (cos32(rlat) * sin32(gcd));
        qcos = geom_clamp1_(qcos);
        brt = acos32(qcos);
    }
    if (dlong < 0.0f)                               // 98
        brt = VC_PI2 - brt;
    out->btr = btr;
    out->btrd = btr * VC_R2D;
    out->brtd = brt * VC_R2D;

    // 104-120: select sample areas by distance
    int km;
    if (gcdkm <= 2000.01f) {
        km = 1;
        out->rd[0] = div32(gcd, 2.0f);
    } else if (gcdkm <= 4000.0f) {
        km = 3;
        out->rd[0] = div32(1000.0f, VC_RZ);
        out->rd[1] = div32(gcd, 2.0f);
        out->rd[2] = gcd - out->rd[0];
    } else {
        km = 5;
        out->rd[0] = div32(1000.0f, VC_RZ);
        out->rd[1] = out->rd[0] + out->rd[0];
        out->rd[2] = div32(gcd, 2.0f);
        out->rd[3] = gcd - out->rd[1];
        out->rd[4] = gcd - out->rd[0];
    }
    out->km = km;

    // 125-184: reflection area coordinates, geomagnetic latitude, ground
    for (int i = 0; i < km; ++i) {                  // DO 180
        float drf = out->rd[i];
        float ctlat = cos32(tlat);
        float rflt, rflg;
        if (ctlat - VC_GEOM_EPSLON < 0.0f) {        // 71,76,76 — STRICT is polar
            rflt = tlat - copysignf(drf, tlat);     // 71
            if (fabsf(rflt) > VC_PIO2)
                rflt = VC_PIO2 * copysignf(1.0f, rflt);
            rflg = rlong;
        } else {                                    // 76
            qcos = cos32(drf) * sin32(tlat)
                 + sin32(drf) * cos32(tlat) * cos32(btr);
            qcos = geom_clamp1_(qcos);
            rflt = VC_PIO2 - acos32(qcos);
            if (cos32(rflt) - VC_GEOM_EPSLON <= 0.0f) {  // 78,78,79: near a pole
                rflg = tlong;                            // 78
            } else {                                     // 79
                qcos = (cos32(drf) - sin32(rflt) * sin32(tlat))
                     / (cos32(rflt) * cos32(tlat));
                qcos = geom_clamp1_(qcos);
                rflg = acos32(qcos);
                if (drf >= VC_PI)
                    rflg = VC_PI2 - rflg;
                rflg = tlong - copysignf(rflg, dlong);
                if (fabsf(rflg) > VC_PI)
                    rflg = rflg - copysignf(VC_PI2, rflg);
            }
        }
        // 155-163: geomagnetic latitude
        qcos = sin32(glt) * sin32(rflt)
             + cos32(glt) * cos32(rflt) * cos32(rflg - glg);
        qcos = geom_clamp1_(qcos);
        float gat = acos32(qcos);
        out->glat[i] = VC_PIO2 - gat;
        out->clat[i] = rflt;
        out->clong[i] = rflg;
        // 166-167: magnetic field variables
        float gyz, gmdip, clg;
        magvar(rflt, rflg, &gyz, &gmdip, &clg);
        out->gyz[i] = gyz;
        out->gmdip[i] = gmdip;
        // 171-182: land-mass lookup -> ground constants
        float rfltd = rflt * VC_R2D;
        float clgd = clg * VC_R2D;
        if (clgd < 0.0f)
            clgd = clgd + 360.0f;
        float wld = noisy(fakp, fakmap, hmym, fakabp, abmap, 7, rfltd, clgd);
        if (wld >= 0.0f) {                          // 115: land
            out->sigpat[i] = 0.001f;
            out->epspat[i] = 4.0f;
        } else {                                    // sea
            out->sigpat[i] = 5.0f;
            out->epspat[i] = 80.0f;
        }
    }

    // 186-187
    float alatd = fabsf(out->clat[0]) * VC_R2D;
    if (km != 1)
        alatd = fabsf(div32(out->clat[0] + out->clat[1] + out->clat[2], 3.0f))
              * VC_R2D;
    out->alatd = alatd;
}

#endif  // VOACAP_GEOM_CUH
