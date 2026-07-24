// PWRDB — transmitter power in dBW at a frequency; the CUDA C twin of
// scalar/pwrdb.py.  The same DO-50 scan GAIN uses, over /cantenna/
// transmitter entries covering the frequency, but returning /pantenna/
// PWRDBA instead of a gain.  The frequency is clamped to 2-30 MHz *for the
// range test only* — never applied to the returned power.
//
// 30 dBW (1 kW) is the default when no transmitter entry matches — unlike
// GAIN, which returns 0 for a miss, this one has a non-zero fallback.
//
// The /cantenna/ + /pantenna/ tables are SHARED, not per-thread, and pass as
// 1-based (A+1,) pointers with A = numants, exactly as in gain.cuh.
// Comparisons and a table read only — no floating-point arithmetic at all.

#ifndef VOACAP_PWRDB_CUH
#define VOACAP_PWRDB_CUH

__device__ inline float pwrdb(const int* iats, const float* xfqs,
                              const float* xfqe, const float* pwrdba,
                              int numants, float freq) {
    float f = freq;
    if (f < 2.0f)
        f = 2.0f;
    if (f > 30.0f)
        f = 30.0f;
    for (int i = 1; i <= numants; ++i) {
        if (iats[i] != 1)
            continue;                   // not the transmitter
        if (xfqs[i] <= f && f <= xfqe[i])
            return pwrdba[i];
    }
    return 30.0f;
}

#endif  // VOACAP_PWRDB_CUH
