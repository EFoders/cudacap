# REAL*4 transcendentals: what the oracle actually computes

Finding from the Module-2 VERSY validation (2026-07-22), established by
instrumenting a verbatim copy of `versy.for` inside the oracle toolchain.

## The three cosines

For REAL*4 `COS(X)` with `X = -0.825022324`, one machine produced two
different answers depending on context:

| context | result bits | value |
|---|---|---|
| GCC compile-time constant folding (MPFR, correctly rounded) | `3F2DB4D5` | 0.678540527820587 |
| glibc runtime `cosf` (Fedora 42, x86-64) | `3F2DB4D6` | 0.678540587425232 |
| our `f32math.cos32` = float32(cos(float64 x)) | `3F2DB4D5` | 0.678540527820587 |

The exact double cosine is 0.67854055578864847 — `D5` is 0.47 ULP below
(correct rounding), `D6` is 0.53 ULP above. glibc's binary32 sinf/cosf
have a ≤1 ULP error bound but are **not correctly rounded**; GCC's MPFR
folding **is**. So:

1. **When the argument is a compile-time constant, gfortran never calls
   libm** — the folded (correctly rounded) value lands in the binary.
   Any flow where the oracle computes trig of runtime data (i.e., all of
   VOACAP proper) uses glibc's answer instead, which differs on a small
   fraction of arguments.
2. **Verification snippets must feed libm runtime values** (e.g. via
   READ), or they measure MPFR, not glibc. (A first standalone check of
   `COS(-0.825022324)` "confirmed" agreement — it had constant-folded.)
3. Our `f32math` helpers produce the correctly-rounded value, i.e. they
   emulate the *folded* behavior, not runtime glibc. Measured mismatch
   rate vs runtime glibc is ~1% of arguments, 1 ULP each.

## Why 1 ULP can matter: VERSY's cancellation

`GOB = COS(CENLAT)` scales every `cos^m(lat)*{cos,sin}(m*lon)` term of
the CCIR geographic series. In the Es coefficient sets, partial sums
swing to ~|660| while the final GAMMA is ~1.7 — a ~350x cancellation
amplification. One ULP in GOB (8.8e-8 relative) becomes ~5e-5 absolute
in GAMMA(1..3). tests/scalar/test_versy.py carries the measured bars
and the full story; GEOM's near-pole ACOS amplification
(tests/scalar/test_geom.py) is the same phenomenon with a different
trigger.

## Decision: characterize the noise floor, don't chase glibc's bits

`f32math` keeps the correctly-rounded (float64-evaluate, round-once)
implementation. We deliberately do **not** reimplement glibc's flt-32
sinf/cosf, for a reason that outranks the 1-ULP mismatch:

- **The production target is the GPU (Stage C, `cuda/`).** There the
  transcendentals are CUDA's, which match neither glibc nor the
  correctly-rounded fallback bit-for-bit. Matching glibc exactly in the
  scalar Stage A leaves would be effort spent on a libm the shipping path
  never runs.
- **What the fidelity bar actually protects is printed output.** VOACAP
  prints MUF to 0.1 MHz, loss/field strength to ~1 dB, reliabilities to
  0.01. The open question is not "does GOB match glibc's last bit" but
  "does a 1-ULP wobble in GOB ever move a printed digit". That is an
  end-to-end property, answered against the tier-A golden decks
  (golden/p2p/matrix), not by per-leaf bit-matching.

So the noise floor is treated as *characterized and bounded*, not
eliminated:

- `golden/dumps/libm/` pins the exact glibc sinf/cosf/acosf/atanf bits
  over a 962-point argument grid (raw REAL*4 intrinsics through the
  oracle) — the empirical measure of the floor, and the ready-made
  validation target **if** a future need forces exact emulation.
- Dump tests on float32 trig-heavy routines (GEOM, VERSY) carry measured
  per-field bars sized by each routine's conditioning, plus
  bit-exact-rate guardrails so the loose bars cannot hide a structural
  regression.

Reviving exact emulation later is a clean transliteration job (glibc's
flt-32 sincosf is a short, fully specified double-precision polynomial;
numpy float64 reproduces it exactly) with the dump already in place to
verify against. It is deferred, not blocked — the caveat to remember is
that x86-64 glibc dispatches ifunc/FMA variants, so any emulation must be
validated against the dump bytes, not the source text.
