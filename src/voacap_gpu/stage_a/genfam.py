"""GENFAM -- frequency dependence of atmospheric noise, plus its decile tables.

ANOIS1 gives the atmospheric noise Fam at 1 MHz for a local-time block; GENFAM
carries it to the operating frequency and supplies the spread statistics that
go with it, all out of the CCIR Report 322 tables REDMAP loads into /TWO/:

* ``FA``   -- Fam at FREQ, from the FAM(14,12) coefficients.  Two nested
  degree-6 Horner evaluations (PZ from FAM 1..7, PX from FAM 8..14), the first
  at the fixed abscissa -0.75 to invert the 1 MHz value into the curve's own
  variable CZ, the second at U = (8*2**log10(FREQ) - 11)/4.
* ``DUA``/``DLA``  -- the upper and lower decile ratios,
* ``DUS``/``DLS``  -- their standard deviations,
* ``DMS``          -- sigma-Fam, the standard deviation of Fam itself,

the last five from quartic polynomials in log10(FREQ) held in DUD(5,12,5).

Two frequency clamps sit in that polynomial loop and both matter:  the decile
curves in Report 322 stop at 20 MHz and sigma-Fam stops at 10 MHz, so X is
pinned at log10(20) above 20 MHz for all five, and again at 1.0 for the fifth
alone above 10 MHz.  The 1993 rewrite comment in genfam.for records that the
previous version had these index-confused, leaving DUA/DLA/DUS/DLS wrong by up
to +-4 dB at 20 MHz -- worth knowing when comparing against older VOACAP output.

IBK selects the table column: local-time block 1..6 in the northern hemisphere,
7..12 in the southern, chosen on the sign of the latitude alone.

float32; ALOG10 and the ``2.**X`` power carry the [[f32-libm-noise]] floor.
"""

from voacap_gpu.stage_a.f32math import F32, log10_32, pow32

_U0 = F32(-0.75)          # the fixed first-pass abscissa


def genfam(st, y2, iblk: int, freq, z):
    """st: IonoState (/TWO/ FAM and DUD); y2: latitude (radians -- only its
    sign is used); iblk: local-time block 1..6; freq: MHz; z: Fam at 1 MHz.

    Returns ``(fa, dua, dla, dms, dus, dls)``, in the Fortran's argument order.
    """
    freq = F32(freq)
    z = F32(z)
    ibk = iblk
    if y2 < 0.0:                      # southern hemisphere -> blocks 7..12
        ibk = ibk + 6
    j = ibk - 1
    fam = st.fam                      # FAM(14,12)
    dud = st.dud                      # DUD(5,12,5)

    def _pzpx(u1):
        """The paired Horner evaluations at abscissa U1 (label 110)."""
        pz = u1 * fam[0, j] + fam[1, j]
        px = u1 * fam[7, j] + fam[8, j]
        for i in range(3, 8):         # DO 115 I=3,7
            pz = u1 * pz + fam[i - 1, j]
            px = u1 * px + fam[i + 6, j]
        return pz, px

    x = log10_32(freq)
    u = (F32(8.0) * pow32(F32(2.0), x) - F32(11.0)) / F32(4.0)
    pz, px = _pzpx(_U0)               # KOP=1: recover the curve variable CZ
    cz = z * pz + px
    cz = z + z - cz
    pz, px = _pzpx(u)                 # KOP=2: evaluate at this frequency
    fa = cz * pz + px

    # Report 322's decile curves end at 20 MHz, sigma-Fam's at 10 MHz.
    if freq > F32(20.0):
        x = log10_32(F32(20.0))
    v = []
    for i in range(1, 6):             # DO 145
        if i == 5 and freq > F32(10.0):
            x = F32(1.0)
        y = dud[0, j, i - 1]
        for jj in range(2, 6):        # DO 140
            y = y * x + dud[jj - 1, j, i - 1]
        v.append(y)
    dua, dla, dus, dls, dms = v
    return fa, dua, dla, dms, dus, dls
