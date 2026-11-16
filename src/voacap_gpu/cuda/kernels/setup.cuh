// SETUP — build a whole CircuitState ON THE DEVICE from raw circuit
// parameters, so the host never runs the Stage A setup chain at all.
//
// WHY THIS HEADER EXISTS
// ----------------------
// ``cuda/driver.py`` already reduced what crosses the bus to the measured
// input set: 3,228 bytes a circuit-hour up, 1,700 back.  That is the whole
// of what LUFFY READS -- but the host still had to BUILD it, and building it
// is ``engine.build_circuit``: REDMAP lookups, GEOM, SANG, and the per-hour
// GEOTIM/VIRTIM/TIMVAR/F2VAR/ESIND/IONSET/CURMUF chain, in Python, per
// circuit-hour.  Measured on the Prop-SL sidecar at a global grid that was
// 97.6% of wall-clock against 0.9% for the kernel and PCIe together: 4.6 ms
// of host prep to feed 42 us of GPU work.  Making the kernel faster could not
// have helped; the kernel was never the problem.
//
// So the chain moves to the device.  Every routine it needs was ported and
// validated bit-exact first (geom, magfin/magvar, sang, geotim/cngtim, versy,
// timvar, ef1var, f2var, esind, ionset, f2dis, curmuf); this header is only
// the COMPOSITION -- the twin of ``engine._install_geometry`` +
// ``engine._setup_hour`` + the state presets ``engine.build_circuit`` applies
// around them.  It contains no physics of its own.
//
// WHAT A CIRCUIT NOW COSTS ON THE BUS
// -----------------------------------
// ``SetupParams`` is 32 bytes: four endpoint coordinates, the hour, the
// long-path flag and two small indices.  Everything else is either uniform
// across the launch (``SetupConfig``, one per distinct SYSTEM card) or a
// shared read-only table already resident for LUFFY's sake.  Against the
// 3,228 bytes ``UPLOAD_FIELDS`` costs, that is 101x less traffic and -- much
// more to the point -- zero Python per circuit-hour.
//
// WHAT IT WRITES, AND WHAT IT DELIBERATELY DOES NOT
// ------------------------------------------------
// Exactly ``driver.UPLOAD_FIELDS`` -- the measured read-before-written set
// plus the two carry fields -- and ``/RAYS/`` NANG + ANG.  Fields OUTSIDE
// that set are left alone on purpose: they are the ECHO and DEAD buckets the
// input-set sweep identified, the kernel never reads them, and zeroing ~1,150
// floats of ionogram scratch per thread that nothing will look at is pure
// cost.  This is the same claim ``LuffyArena`` already relies on when it
// declines to re-zero between submissions, so the device-setup path inherits
// its proof rather than making a new one.
//
// The consequence for testing: a device-built CircuitState is NOT byte-equal
// to a host-built one, and must not be compared as though it were.  It is
// equal on ``UPLOAD_FIELDS``, and it produces the same LUFFY answer.  Both
// are what ``tests/cuda/test_setup_batch.py`` asserts.
//
// THREE THINGS PINNED ELSEWHERE THAT THIS COMPOSITION RELIES ON
// -------------------------------------------------------------
// * ``rays.ang`` is a STATIC blkdat table (raysstate.py's dataclass default,
//   the same forty numbers for every circuit), so ``sang_rays`` fills it from
//   the device constant and nothing per-circuit is uploaded for it.
// * VIRTIM's ``ab`` row depends only on GMT, and on the ITIM >= 0 arm GMT is
//   the requested hour exactly -- so the host precomputes 25 rows per
//   ionosphere (hours 0..24) and every circuit indexes one.  319 floats x 25
//   is 32 KB per ionosphere against 1,276 bytes PER CIRCUIT-HOUR if it were
//   carried per thread.  ITIM < 0 (the TIME card in local time at the
//   transmitter) would make GMT a derived quantity and break that; the driver
//   refuses it rather than computing a wrong row.
// * ``clat`` / ``clong`` / ``gmdip`` / ``rd`` are NOT in the input set: the
//   setup chain is their only consumer.  They are written into the state here
//   because the chain reads them back out of it, but nothing downstream does.

#ifndef VOACAP_SETUP_CUH
#define VOACAP_SETUP_CUH

#include "circuit.cuh"
#include "constants.cuh"
#include "curmuf.cuh"
#include "esind.cuh"
#include "f2var.cuh"
#include "geom.cuh"
#include "geotim.cuh"
#include "ionset.cuh"
#include "raysview.cuh"
#include "sang.cuh"
#include "timvar.cuh"

// One VIRTIM row, and the hours 0..24 the host precomputes per ionosphere.
#define VC_SETUP_NAB   319
#define VC_SETUP_NHOUR 25

// THE HOUR GUARD.  A legal hour is 1..24.  Anything else has no VIRTIM row
// (`ab` holds rows 0..24 per ionosphere, so hour 60 reads another table or
// past the end) and overruns TimeState's 25-long UTIME/XLMT, so the build is
// REFUSED rather than run on garbage.  The refusal travels as IPFLAG
// VC_IPFLAG_REFUSED -- no legal IPFLAG is negative -- and luffy_batch turns
// it into status VC_STATUS_TABLE_OOB, which is VC_ST_SETUP_HOUR_OOB (1301) in
// the global table (status.cuh): Stage A raises IndexError for hour 25 for the
// same reason.  driver.py range-checks on the host first; this is the defence in
// depth for a caller that bypasses it.
#ifndef VC_IPFLAG_REFUSED
#define VC_IPFLAG_REFUSED   (-1)
#endif
#include "status.cuh"
#ifndef VC_STATUS_TABLE_OOB
#define VC_STATUS_TABLE_OOB VC_ST_SETUP_HOUR_OOB
#endif
#define VC_SETUP_HOUR_MIN 1
#define VC_SETUP_HOUR_MAX (VC_SETUP_NHOUR - 1)

// Per-ionosphere strides for the tables only the setup chain needs (the ones
// LUFFY also uses are in circuit.cuh).
#define VC_IKIM_STRIDE (10 * 6)
#define VC_IKIM_NCOLS  6
#define VC_F2D_STRIDE  (16 * 6 * 6)
#define VC_ANEW_STRIDE 3
#define VC_ACHI_STRIDE 2

// blkdat.for:163 GCDLNG — the long-path threshold IPFLAG turns on.
#define VC_SETUP_GCDLNG 10000.0f

// ---------------------------------------------------------------------------
// The three input blocks
// ---------------------------------------------------------------------------

// PER CIRCUIT-HOUR — 32 bytes, and the only thing that scales with N.
// Endpoint coordinates in degrees (south/west negative); `hour` is the 1-based
// UTC hour; `npsl` is GEOM's long-path flag; `iono` and `cfg` index the
// shared ionosphere and SYSTEM-card blocks.
struct SetupParams {
    float tlatd, tlongd, rlatd, rlongd;
    int hour;
    int npsl;
    int iono;
    int cfg;
};

// PER SYSTEM CARD — one for a whole Prop-SL request, usually one for a whole
// launch.  These are the values ``engine.build_circuit`` copies into the
// Circuit before the chain runs; carrying them here rather than per circuit
// is what keeps SetupParams at 32 bytes.
struct SetupConfig {
    float pole_lat, pole_lon;   // geomagnetic north pole (engine.POLE_DEFAULT)
    float amind;                // SANG minimum take-off angle, degrees
    float rsn, dmp, pmp;        // required S/N, max spread dB, max delay ms
    float psc[5];               // /PSCA/ FPROB multipliers, 1-based
    int lufp, method, noise, mspec;
    int nyear_is_2089, want_service_prob, info, itim;
    float frel[30];             // the user frequency list; [12] is overwritten
};

// THE SHARED READ-ONLY TABLES, one block per ionosphere.  Five of them
// (fakp/fakmap/hmym/fakabp/abmap) are the REDMAP overlays LUFFY already has
// resident; the rest are the setup chain's own.
struct SetupTables {
    const float* ab;            // (niono * 25, 319) VIRTIM rows, hours 0..24
    const int*   ikim;          // (niono, 10, 6)
    const float* anew;          // (niono, 3)
    const float* bnew;          // (niono, 3)
    const float* achi;          // (niono, 2)
    const float* bchi;          // (niono, 2)
    const float* fakp;          // (niono, 29, 16, 6)
    const float* fakmap;        // (niono, 29, 16)
    const float* hmym;          // (niono, 29, 16)
    const float* fakabp;        // (niono, 2, 6)
    const float* abmap;         // (niono, 2, 3)
    const float* f2d;           // (niono, 16, 6, 6)
    const int*   month;         // (niono,)
    const float* ssn;           // (niono,)
};

// ---------------------------------------------------------------------------
// IPFLAG — engine._ipflag, transcribed
// ---------------------------------------------------------------------------
// hfmufs.for's system-performance arm.  `method`/`mspec` are the INTERNAL
// pair engine._resolve_method produces (METHOD 30 arrives as 20/121), which
// is why `meth` is recovered from MSPEC before the METHOD 30 test.
__device__ __forceinline__ int setup_ipflag(int method, int mspec,
                                            float gcdkm) {
    int meth = (mspec == 121) ? 30 : method;
    if (method == 22 || method == 25 || mspec == 125)
        return 100;
    int ipfg = 100;
    if (mspec != 121 && gcdkm >= VC_SETUP_GCDLNG)
        ipfg = 200;
    if (meth == 30 && gcdkm >= VC_SETUP_GCDLNG)
        ipfg = 200;
    return ipfg;
}

// ---------------------------------------------------------------------------
// The presets build_circuit applies to a fresh Circuit
// ---------------------------------------------------------------------------

// /SON/ as a fresh SonState: every number zero, every label "".
//
// The struct is padding-free and every member is four bytes (state.cuh's ABI
// contract, pinned by tests/cuda/test_state.py), so the zeroing is a word
// loop over the whole block; the three label arrays are then set to -1, which
// is what state.cuh's LAYTYP encoding and circuit.cuh's VC_MDL_EMPTY both use
// for the empty label.
//
// /SON/ is in CARRY_FIELDS, not INPUT_FIELDS: LUFFY skips every frequency
// slot whose FREL is zero and leaves whatever was there, so a reused device
// buffer would hand back the PREVIOUS batch's numbers in the unused columns.
// That is the Fortran COMMON persistence the replay depends on and it is
// correct for a Circuit reused across hours -- but a circuit built from raw
// parameters is a FRESH one, so the preset has to be applied here.
__device__ inline void setup_reset_son(SonState* son) {
    int* w = (int*)son;
    for (int k = 0; k < (int)(sizeof(SonState) / 4); ++k)
        w[k] = 0;
    for (int k = 0; k < 14; ++k) {
        son->mdl[k] = VC_MDL_EMPTY;
        son->mode[k] = -1;
        son->moder[k] = -1;
    }
}

// /MUFS/ as a fresh MufState.  CURMUF writes the nine hour ladders at TIME.IT
// only and the five layer arrays entire, so the other 24 hours keep their
// entry values -- zero on a fresh Circuit.  XLUF is SETOUT's -1 preset, which
// nothing on the system-performance arms writes and which therefore has to
// survive all the way back to the host (hence CARRY_FIELDS).
__device__ inline void setup_reset_muf(MufState* muf) {
    for (int k = 0; k < 25; ++k) {
        muf->emuf[k] = 0.0f;
        muf->f1muf[k] = 0.0f;
        muf->f2muf[k] = 0.0f;
        muf->esmuf[k] = 0.0f;
        muf->allmuf[k] = 0.0f;
        muf->fot[k] = 0.0f;
        muf->hpf[k] = 0.0f;
        muf->angmuf[k] = 0.0f;
        muf->xluf[k] = -1.0f;               // SETOUT's preset
    }
    // AFMUF is read by INMUF/SETTXR and written by NOTHING in the port, so a
    // caller who does not set it is asking for zeros; the other four layer
    // arrays CURMUF fills, but they are cheap and this keeps the block equal
    // to a fresh MufState field for field.
    for (int k = 0; k < 5; ++k) {
        muf->sigl[k] = 0.0f;
        muf->sigu[k] = 0.0f;
        muf->delmuf[k] = 0.0f;
        muf->hpmuf[k] = 0.0f;
        muf->htmuf[k] = 0.0f;
        muf->fvmuf[k] = 0.0f;
        muf->afmuf[k] = 0.0f;
        muf->nhopmf[k] = 0;
        muf->yfot[k] = 0.0f;
        muf->yhpf[k] = 0.0f;
        muf->ymuf[k] = 0.0f;
    }
    muf->modmuf = 0;
}

// ---------------------------------------------------------------------------
// engine._install_geometry
// ---------------------------------------------------------------------------
// GEOM's result plus the loose /DON/ scalars into the state.  Stage A writes
// sample areas 1..KM and leaves the rest at the dataclass zero; GeomResult
// already zeroes its entries past KM, so all five slots copy unconditionally
// and the state matches a fresh Circuit either way.
//
// The two longitude tests are NOT symmetric and that is deliberate
// (engine.py:376-377): the transmitter is tested on the caller's TLATD, the
// receiver on GEOM's RLATD -- which GEOM may have NUDGED when the endpoints
// were within DDX of each other.
__device__ inline void setup_install_geometry(CircuitState* cs,
                                              const GeomResult& g,
                                              float tlatd, float tlongd,
                                              float rlongd, float amind) {
    LayerState* lay = &cs->layers;
    lay->km = g.km;
    lay->iedp = -1;
    lay->clat[0] = 0.0f;
    lay->clong[0] = 0.0f;
    lay->glat[0] = 0.0f;
    lay->rd[0] = 0.0f;
    lay->gyz[0] = 0.0f;
    lay->gmdip[0] = 0.0f;
    lay->sigpat[0] = 0.0f;
    lay->epspat[0] = 0.0f;
    for (int k = 0; k < 5; ++k) {
        lay->clat[k + 1] = g.clat[k];
        lay->clong[k + 1] = g.clong[k];
        lay->glat[k + 1] = g.glat[k];
        lay->rd[k + 1] = g.rd[k];
        lay->gyz[k + 1] = g.gyz[k];
        lay->gmdip[k + 1] = g.gmdip[k];
        lay->sigpat[k + 1] = g.sigpat[k];
        lay->epspat[k + 1] = g.epspat[k];
    }
    cs->gcd = g.gcd;
    cs->gcdkm = g.gcdkm;
    cs->amind = amind;
    // /DON/ endpoint coordinates in radians, with GEOM's near-pole longitude
    // zeroing (geom.for:52-53).
    cs->tlong = (fabsf(tlatd) > 89.9f) ? 0.0f : tlongd * VC_D2R;
    cs->rlong = (fabsf(g.rlatd) > 89.9f) ? 0.0f : rlongd * VC_D2R;
    cs->rlongd = rlongd;
    cs->rlat = g.rlatd * VC_D2R;
}

// ---------------------------------------------------------------------------
// engine._setup_hour
// ---------------------------------------------------------------------------
// One prediction hour's ionosphere and MUF build, in exactly the order
// hfmufs.for drives it.  VIRTIM is the one link that is NOT called here: its
// row depends only on GMT, so the host precomputed it and `ab` points at the
// right one.  ``tests/cuda/test_setup_chain.py`` is this sequence's
// acceptance test against ``engine._setup_hour``, field for field.
__device__ inline void setup_hour(CircuitState* cs, const SetupTables& t,
                                  int io, int hour) {
    LayerState* lay = &cs->layers;
    const float* ab = t.ab + ((size_t)io * VC_SETUP_NHOUR + hour)
                             * VC_SETUP_NAB;
    const int* ikim = t.ikim + (size_t)io * VC_IKIM_STRIDE;

    geotim(lay, &cs->time, hour, cs->tlong, cs->rlong);
    timvar(lay, ikim, VC_IKIM_NCOLS, ab,
           t.anew + (size_t)io * VC_ANEW_STRIDE,
           t.bnew + (size_t)io * VC_ANEW_STRIDE,
           t.achi + (size_t)io * VC_ACHI_STRIDE,
           t.bchi + (size_t)io * VC_ACHI_STRIDE,
           t.fakp + (size_t)io * VC_FAKP_STRIDE,
           t.fakmap + (size_t)io * VC_FAKMAP_STRIDE,
           t.hmym + (size_t)io * VC_HMYM_STRIDE,
           t.fakabp + (size_t)io * VC_FAKABP_STRIDE,
           t.abmap + (size_t)io * VC_ABMAP_STRIDE,
           cs->time.gmt);
    f2var(lay, ikim, VC_IKIM_NCOLS, ab);
    esind(lay, ikim, VC_IKIM_NCOLS, ab);
    ionset(lay);
    curmuf(lay, &cs->muf, t.f2d + (size_t)io * VC_F2D_STRIDE,
           cs->gcdkm, cs->gcd, cs->amind, lay->ssn, cs->time.it, lay->iedp);
    cs->frq.frel[12] = cs->muf.allmuf[cs->time.it];
}

// ---------------------------------------------------------------------------
// The whole thing
// ---------------------------------------------------------------------------
// One circuit-hour, from raw parameters to a CircuitState LUFFY can read.
// Returns IPFLAG; writes the three per-hour MUF summary numbers the host
// reports separately (``engine.predict`` reads ALLMUF/FOT/HPF at TIME.IT off
// the built circuit, and with the build on the device there is nothing on the
// host left to read them from).
__device__ inline int setup_circuit(CircuitState* cs, const RaysView& rays,
                                    const SetupParams& p,
                                    const SetupConfig& c,
                                    const SetupTables& t,
                                    float* muf_out) {
    LayerState* lay = &cs->layers;

    // --- the hour guard (see VC_IPFLAG_REFUSED) ----------------------------
    // /SON/ is reset so the refused row reads as a fresh, empty answer rather
    // than the previous occupant's; LUFFY never runs on it.
    if (p.hour < VC_SETUP_HOUR_MIN || p.hour > VC_SETUP_HOUR_MAX) {
        setup_reset_son(&cs->son);
        muf_out[0] = muf_out[1] = muf_out[2] = __int_as_float(0x7fc00000);
        return VC_IPFLAG_REFUSED;
    }

    // --- GEOM, then the geometry install ----------------------------------
    GeomResult g;
    geom(p.tlatd, p.tlongd, p.rlatd, p.rlongd, p.npsl, c.pole_lat, c.pole_lon,
         t.fakp + (size_t)p.iono * VC_FAKP_STRIDE,
         t.fakmap + (size_t)p.iono * VC_FAKMAP_STRIDE,
         t.hmym + (size_t)p.iono * VC_HMYM_STRIDE,
         t.fakabp + (size_t)p.iono * VC_FAKABP_STRIDE,
         t.abmap + (size_t)p.iono * VC_ABMAP_STRIDE, &g);
    setup_install_geometry(cs, g, p.tlatd, p.tlongd, p.rlongd, c.amind);

    // --- SANG: NANG into /RAYS/, plus the static blkdat ANG table ----------
    sang_rays(rays, g.gcdkm, c.amind);

    // --- the fresh-Circuit presets ----------------------------------------
    lay->month = t.month[p.iono];
    lay->ssn = t.ssn[p.iono];
    for (int k = 0; k < 5; ++k)
        lay->psc[k] = c.psc[k];
    // /ES/ FS and IELECT are read (esmod/regmod, lecden) and written by
    // nothing in the port -- inputs for a caller who sets them, zero here.
    for (int k = 0; k < 4 * 6; ++k)
        lay->fs[k] = 0.0f;
    for (int k = 0; k < 4; ++k)
        lay->ielect[k] = 0;
    setup_reset_muf(&cs->muf);
    setup_reset_son(&cs->son);
    for (int k = 0; k < 30; ++k)
        cs->frq.frel[k] = c.frel[k];

    // blkdat.for:163 + :183 — /LPATH/'s four constants and LTXRGM.  DELOPT,
    // GMIN and YMIN are measured inputs (SELTXR, GETTOP, LNGPAT read them);
    // LTXRGM's 1,1 matters because LNGPAT's guard is ``LTXRGM(n) <= 0``, so a
    // never-run SELTXR must read as "ray 1 at both ends".
    cs->lpath.gcdlng = VC_SETUP_GCDLNG;
    cs->lpath.delopt = 3.0f;
    cs->lpath.gmin = 3.0f;
    cs->lpath.ymin = 0.1f;
    cs->lpath.ltxrgm[0] = 0;
    cs->lpath.ltxrgm[1] = 1;
    cs->lpath.ltxrgm[2] = 1;

    cs->rsn = c.rsn;
    cs->dmp = c.dmp;
    cs->pmp = c.pmp;
    cs->lufp = c.lufp;
    cs->method = c.method;
    cs->noise = c.noise;
    cs->mspec = c.mspec;
    // JLONG is LUFFY's output-only short/long-path tag and is measured DEAD as
    // an input; -1 is nonetheless the Circuit dataclass default, so matching it
    // keeps a device-built state equal to a host-built one on the whole loose
    // scalar block rather than equal-except-one.
    cs->jlong = -1;
    cs->nyear_is_2089 = c.nyear_is_2089;
    cs->want_service_prob = c.want_service_prob;
    cs->info = c.info;
    cs->time.itim = c.itim;

    // --- the hour ----------------------------------------------------------
    setup_hour(cs, t, p.iono, p.hour);

    int it = cs->time.it;
    muf_out[0] = cs->muf.allmuf[it];
    muf_out[1] = cs->muf.fot[it];
    muf_out[2] = cs->muf.hpf[it];
    return setup_ipflag(c.method, c.mspec, cs->gcdkm);
}

#endif  // VOACAP_SETUP_CUH
