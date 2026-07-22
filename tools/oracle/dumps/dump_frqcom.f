      PROGRAM DUMPFRQCOM
C     Tier-B oracle dump for FRQCOM + SETLUF, the driver's two ends of the
C     frequency list.  HFMUFS builds the list with FRQCOM before LUFFY and
C     reads it back with SETLUF after, so the driver runs them in that order
C     here too: FRQCOM's FREL IS SETLUF's input, and driving them together
C     pins the handoff.  Both routines' outputs are dumped separately, so a
C     failure still localises to one of them.
C
C     The FREL slots after FRQCOM are dumped as QFREL, and the LUF slot is
C     dumped TWICE -- once as QFREL13 (what FRQCOM left, which is a sentinel
C     in the case-1 arm because that arm alone does not zero it) and once as
C     LUF13 (what SETLUF wrote).
C
C     NOT GENERATED: case 3 with a lower MUF far above the highest usable
C     frequency.  NE is (XFL-FLOW)/DELF truncated with nothing bounding it,
C     and the Fortran then writes FREA(NE+1) past the end of the common.
C     The Python port raises there; the generator filters those cases out so
C     the oracle is never asked to corrupt itself.
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON / ION / IHRE, IHRO, IHRS, LUFP, METHOD, NOISE, NPSL
      COMMON/MUFS/EMUF(24),F1MUF(24),F2MUF(24),ESMUF(24),ALLMUF(24),FOT
     A (24),XLUF(24),HPF(24),ANGMUF(24),MODMUF,SIGL(4),SIGU(4),DELMUF(4)
     B ,HPMUF(4),HTMUF(4),FVMUF(4),AFMUF(4),NHOPMF(4),YFOT(4),YHPF(4)
     C ,YMUF(4)
      COMMON / SON / ANGLE(13), ANGLER(13), CPROB(13), DBLOS(13),
     A DBLOSL(13), DBLOSU(13), DBU(13), DELAY(13), DBW(13), NHP(13),
     B XNYNOIS(13), PROBMP(13), RELIAB(13), SNDB(13), SNPR(13),
     C SNRLW(13), SNRUP(13), SPROB(13), VHIGH(13), RNEFF(13),MDL(13)
      CHARACTER MDL*1
      COMMON / TIME / IT, GMT, UTIME(24), GMTR, XLMT(24), ITIM, JTX
      CHARACTER HEAD*600
      DIMENSION FRELI(11), RELI(11), QFREL(13)
      OPEN(10,FILE='frqcom_cases.csv',STATUS='OLD')
      OPEN(11,FILE='frqcom_dump.csv')
C          ---- CSV header ----
      HEAD=' '
      DO 11 I=1,13
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A,I2.2)') ',qfrel',I
      WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A)') ',xluf,luf13'
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one case per record (list-directed, 31 values) ----
20    READ(10,*,END=99) ITI,IFREQ,ALMUFI,FOTI,EMUFI,F2MUFI,HPFI,LUFPI,
     A  FRELI,FR13I,RELI
      IT     = ITI
      LUFP   = LUFPI
      ALLMUF(IT) = ALMUFI
      FOT(IT)    = FOTI
      EMUF(IT)   = EMUFI
      F2MUF(IT)  = F2MUFI
      HPF(IT)    = HPFI
      XLUF(IT)   = -5555.
      DO 21 I=1,29
21    FREL(I) = -4444.
      DO 22 I=1,11
      FREL(I)   = FRELI(I)
22    RELIAB(I) = RELI(I)
      FREL(12) = -4444.
      FREL(13) = FR13I
C          ---- HFMUFS's own dispatch, verbatim ----
      IF(IFREQ.le.0) then
         CALL FRQCOM(FREL,IFREQ)
      else
         FREL(12) = ALLMUF(IT)
      end if
      DO 23 I=1,13
23    QFREL(I) = FREL(I)
      CALL SETLUF
      WRITE(11,30) (QFREL(I),I=1,13),XLUF(IT),FREL(13)
      GO TO 20
30    FORMAT(ES18.9E3,14(',',ES18.9E3))
99    CLOSE(10)
      CLOSE(11)
      END
