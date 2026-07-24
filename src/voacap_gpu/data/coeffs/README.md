# Bundled CCIR/URSI ionospheric coefficients

The 12 monthly `coeffNN.asc` sets and the `fof2CCIR.asc` / `fof2URSI.asc`
atlases, vendored from voacapl's `itshfbc/coeffs/` so `voacap_gpu` is
self-contained (predict() reads them at runtime via `voacap_gpu.coeffs`).

Public domain: original VOACAP is a US Government (NTIA/ITS) work; the
voacapl distribution is CC0. Byte-identical to the oracle sources, checked by
`tests/test_coeffs.py`.
