// gridreduce.cuh -- PropServer's per-cell `cellAgg`, on the device.
//
// This is gridjob.CellReducer (itself a transcription of models/
// voacap_grid.go's cellAgg) run where the samples already are, so a grid
// sweep never has to bring 2.8 million (hour, frequency) samples across the
// bus to throw all but 16,471 of them away.
//
// IT IS A SEQUENTIAL SCAN PER CELL, ON PURPOSE.  The reduction is
// ORDER-DEPENDENT -- a new best RESETS the hourly accumulator, a tie keeps the
// incumbent (strict `>`), and only later samples at the winning frequency
// average -- so each thread owns ONE cell and replays that cell's samples in
// exactly the sweep order the host reducer sees: increasing circuit row g
// (hours -> frequency-chunk -> cell, so a cell's rows are g = cell + b*n_cells
// for block b), then frequency j within the row.  Cells are independent, so
// the parallelism is across cells and nothing is reordered within one.
//
// Chunks arrive one launch at a time, and the driver chains the launches
// chunk-to-chunk with events, so the scan also crosses chunk boundaries in
// order.  The per-cell state lives in device arrays between launches.
//
// ARITHMETIC.  Everything is binary64, as the host reducer is: the sample is
// the binary32 /SON/ word widened (exact), optionally sanitised the way the
// Prop-SL service's `_finite` does (non-finite -> -200 dB, reliability -> 0),
// then the reliability scaled (`rel * rel_scale`, one correctly rounded
// multiply, as numpy does), then compared and summed.  The adds, the multiply
// and the final divide are written as __dadd_rn / __dmul_rn / __ddiv_rn so
// there is no FMA contraction to argue about (compile.py also pins
// --fmad=false); `sum += x` in sample order is then the same float64 sequence
// as numpy's `s[ci] += x` per feed call.
//
// The non-finite rule is CellReducer.feed's extension: a sample whose score or
// any of its four values is NaN / +-inf after the transform contributes
// nothing, exactly like a failed circuit.  A failed circuit (status != 0)
// contributes nothing at all, every frequency of it.

#ifndef VOACAP_GRIDREDUCE_CUH
#define VOACAP_GRIDREDUCE_CUH

#define VC_GR_MAXF 11            // frequencies per circuit (the FREL capacity)
#define VC_GR_SENTINEL_DB  (-200.0)
#define VC_GR_SENTINEL_REL (0.0)

__device__ __forceinline__ bool gr_finite(double x) {
    return isfinite(x);                 // numpy.isfinite: not NaN, not +-inf
}

__device__ __forceinline__ double gr_word(const int* row, int off) {
    return (double)__int_as_float(row[off]);
}

// One chunk of circuit rows [lo, lo + c) in GLOBAL sweep numbering.  Local row
// r of the chunk is global row lo + r.  `circ` is the chunk's state as int32
// words, `stride` words per row; the four sample columns start at word
// offsets off_* (slot 0) and frequency j of the row lives in slot slot0 + j.
extern "C" __global__ void grid_reduce_chunk(
        const int* __restrict__ circ, int stride,
        int off_snr, int off_dbw, int off_dbu, int off_rel,
        const int* __restrict__ status,
        long long lo, int c,
        int n_cells, int n_fchunks,
        const double* __restrict__ fq,      // [n_fchunks * VC_GR_MAXF]
        const int* __restrict__ nfq,        // [n_fchunks]
        int slot0, int goal, int monthly, int sanitize, double rel_scale,
        int* has, double* best,
        double* b_snr, double* b_dbw, double* b_dbu, double* b_rel,
        double* b_freq, long long* count,
        double* s_snr, double* s_dbw, double* s_dbu, double* s_rel) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    long long hi = lo + (long long)c;
    long long n = (long long)n_cells;
    long long first = lo + ((cell - lo % n) % n + n) % n;
    if (first >= hi) return;             // this chunk has no row of this cell

    int h = has[cell];
    double bs = best[cell];
    double v_snr = b_snr[cell], v_dbw = b_dbw[cell];
    double v_dbu = b_dbu[cell], v_rel = b_rel[cell], v_f = b_freq[cell];
    long long cnt = count[cell];
    double a_snr = s_snr[cell], a_dbw = s_dbw[cell];
    double a_dbu = s_dbu[cell], a_rel = s_rel[cell];

    for (long long g = first; g < hi; g += n) {
        int r = (int)(g - lo);
        if (status[r] != 0) continue;            // a failed circuit: nothing
        int ci = (int)((g / n) % (long long)n_fchunks);
        int nf = nfq[ci];
        const int* row = circ + (long long)r * stride;
        for (int j = 0; j < nf; ++j) {
            int k = slot0 + j;
            double snr = gr_word(row, off_snr + k);
            double dbw = gr_word(row, off_dbw + k);
            double dbu = gr_word(row, off_dbu + k);
            double rel = gr_word(row, off_rel + k);
            if (sanitize) {                      // the service's _finite
                if (!gr_finite(snr)) snr = VC_GR_SENTINEL_DB;
                if (!gr_finite(dbw)) dbw = VC_GR_SENTINEL_DB;
                if (!gr_finite(dbu)) dbu = VC_GR_SENTINEL_DB;
                if (!gr_finite(rel)) rel = VC_GR_SENTINEL_REL;
            }
            rel = __dmul_rn(rel, rel_scale);
            double score = goal == 0 ? snr : goal == 1 ? dbw
                         : goal == 2 ? dbu : rel;
            if (!(gr_finite(score) && gr_finite(snr) && gr_finite(dbw)
                  && gr_finite(dbu) && gr_finite(rel)))
                continue;                        // CellReducer's NaN rule
            double f = fq[ci * VC_GR_MAXF + j];
            if (!h || score > bs) {              // strictly: ties keep the incumbent
                h = 1;
                bs = score;
                v_snr = snr; v_dbw = dbw; v_dbu = dbu; v_rel = rel; v_f = f;
                if (monthly) {                   // RESTART the accumulator
                    a_snr = snr; a_dbw = dbw; a_dbu = dbu; a_rel = rel;
                    cnt = 1;
                }
            } else if (monthly && v_f == f) {    // only the winning frequency
                a_snr = __dadd_rn(a_snr, snr);
                a_dbw = __dadd_rn(a_dbw, dbw);
                a_dbu = __dadd_rn(a_dbu, dbu);
                a_rel = __dadd_rn(a_rel, rel);
                cnt += 1;
            }
        }
    }

    has[cell] = h;
    best[cell] = bs;
    b_snr[cell] = v_snr; b_dbw[cell] = v_dbw; b_dbu[cell] = v_dbu;
    b_rel[cell] = v_rel; b_freq[cell] = v_f;
    count[cell] = cnt;
    s_snr[cell] = a_snr; s_dbw[cell] = a_dbw;
    s_dbu[cell] = a_dbu; s_rel[cell] = a_rel;
}

// cellAgg.finalise for every cell -> out[5 * n_cells]: snr, dbw, dbu, rel,
// freq as five consecutive columns.  A cell whose winning frequency recurred
// gets the mean (sum / count, one correctly rounded divide); a cell that never
// saw a usable sample gets PropServer's sentinels and the default frequency.
extern "C" __global__ void grid_finalise(
        const int* has,
        const double* b_snr, const double* b_dbw, const double* b_dbu,
        const double* b_rel, const double* b_freq, const long long* count,
        const double* s_snr, const double* s_dbw, const double* s_dbu,
        const double* s_rel, double default_freq, double* out, int n_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    double snr = b_snr[i], dbw = b_dbw[i], dbu = b_dbu[i], rel = b_rel[i];
    double freq = b_freq[i];
    long long cnt = count[i];
    if (cnt > 1) {
        double d = (double)cnt;
        snr = __ddiv_rn(s_snr[i], d);
        dbw = __ddiv_rn(s_dbw[i], d);
        dbu = __ddiv_rn(s_dbu[i], d);
        rel = __ddiv_rn(s_rel[i], d);
    }
    if (!has[i]) {
        snr = VC_GR_SENTINEL_DB; dbw = VC_GR_SENTINEL_DB;
        dbu = VC_GR_SENTINEL_DB; rel = VC_GR_SENTINEL_REL;
        freq = default_freq;
    }
    long long n = (long long)n_cells;
    out[i] = snr;
    out[n + i] = dbw;
    out[2 * n + i] = dbu;
    out[3 * n + i] = rel;
    out[4 * n + i] = freq;
}

#endif  // VOACAP_GRIDREDUCE_CUH
