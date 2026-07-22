"""ESREG -- would compute a mixed Es-F mode.  It does not.  It presets.

esreg.for is 363 lines.  Twenty-six of them run.  After presetting /ZON/
slots 6 and 7 to the no-mode markers the routine hits an unconditional
``RETURN``, and the source says why in as many words::

    C
    C  ERRORS IN THE CODE BELOW
    C
          RETURN
    c
    c--ESREG-------------------------------Nothing DONe here!!!
    c                                       due to RETURN above

Everything past that point -- the sample-area scan, the FDIST-style
reflectrix search, the split Es-then-F group path, the two absorption terms,
the four PRBMUF calls and the decile adjustments -- is unreachable, and by
the authors' own note wrong.  There is no ENTRY statement and nothing
branches into it from outside, so the dead region has no observable effect
whatsoever.

It is **not ported**, and that is the point of this module existing at all
rather than the caller simply inlining twenty assignments: a reader who finds
"ESREG" in the call graph and goes looking for the mixed-mode physics should
land here and learn that VOACAP has never computed it.

The intent, for the record, was to add radiation-angle possibilities by
taking one Es hop and the remainder F hops -- angles that neither REGMOD nor
ESMOD can produce, since each is confined to a single reflecting layer.

Note the presets are NOT the same as ESMOD's for slots 4 and 5: CREL is
+1000 here and -1000 there.  Transliterated as found.
"""

from voacap_gpu.stage_a.f32math import F32

ESREG_SLOTS = (6, 7)        # IMS = 6, IME = 7


def esreg(zon) -> None:
    """Preset /ZON/ slots 6 and 7 to the no-mode markers.  Mutates `zon`.

    This is the whole of ESREG's reachable behaviour; see the module
    docstring.
    """
    for im in ESREG_SLOTS:              # DO 100
        zon.abps[im] = F32(1000.0)
        zon.crel[im] = F32(1000.0)
        zon.eff[im] = F32(0.0)
        zon.fldst[im] = F32(-1000.0)
        zon.grlos[im] = F32(1000.0)
        zon.hn[im] = F32(-1.0)
        zon.hp[im] = F32(-1.0)
        zon.prob[im] = F32(0.001)
        zon.rely[im] = F32(0.001)
        zon.rgain[im] = F32(0.0)
        zon.tgain[im] = F32(0.0)
        zon.timed[im] = F32(-1.0)
        zon.tloss[im] = F32(1000.0)
        zon.b[im] = F32(-1.0)
        zon.fslos[im] = F32(1000.0)
        zon.adv[im] = F32(1000.0)
        zon.obf[im] = F32(1000.0)
        zon.nmode[im] = 5
        zon.tllow[im] = F32(10.0)
        zon.tlhgh[im] = F32(10.0)
        zon.sigpow[im] = F32(-1000.0)
        zon.sn[im] = F32(-1000.0)
        zon.spro[im] = F32(0.001)
