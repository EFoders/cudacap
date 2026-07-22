      PROGRAM DUMPSERPRB
C     Tier-B oracle dump for SERPRB + MPATH (service and multipath
C     probability), driven together in the order LUFFY calls them.
C
C     Both read /allMODE/ and write one /SON/ column, and MPATH depends on
C     NREL, which RELBIL sets -- so NREL is injected here rather than being
C     produced, keeping the two routines independent of RELBIL's selection.
C
C     NMMOD is never zero in these cases.  With no modes SERPRB's two loops
C     both skip, IMAX stays 1, and D10S/D50S come from uninitialised local
C     arrays; the Python port raises there rather than reproducing that, so
C     there is nothing to compare.
      COMMON/ANOIS/ATNU,ATNY,CC,TM,RCNSE,DU,DL,SIGM,SYGU,SYGL,KJ,JK
      COMMON / CON / D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON / ION / IHRE, IHRO, IHRS, LUFP, METHOD, NOISE, NPSL
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON / SON / ANGLE(13), ANGLER(13), CPROB(13), DBLOS(13),
     A DBLOSL(13), DBLOSU(13), DBU(13), DELAY(13), DBW(13), NHP(13),
     B XNYNOIS(13), PROBMP(13), RELIAB(13), SNDB(13), SNPR(13),
     C SNRLW(13), SNRUP(13), SPROB(13), VHIGH(13), RNEFF(13),MDL(13)
      CHARACTER MDL*1
      COMMON/TON/ADJ,ADS,GNOS,GOT,REL,SL,SLS,SPR,SU,SUS
     A ,XNOISE,ZNOISE,NF
      COMMON / allMODE /ABPS(20),CREL(20),FLDST(20),HN(20),HP(20),
     1PROB(20),RELY(20),RGAIN(20),SIGPOW(20),SN(20),
     2SPRO(20),TGAIN(20),TIMED(20),TLOSS(20),B(20),FSLOS(20),
     3grlos(20),adv(20),obf(20),
     CNMODE(20),TLLOW(20),TLHGH(20),EFF(20),NREL,NMMOD
      COMMON / FILES / LUI, LUO, LU2, LU5, LU6, LU15, LU16,
     1                 LU20, LU25, LU26, LU35
      CHARACTER HEAD*1200
      DIMENSION HPI(20),SNI(20),TLLOWI(20),TLHGHI(20),SIGPWI(20),
     +          TIMEDI(20),RELYI(20)
      DATA DCL/1.28/
      OPEN(10,FILE='serprb_cases.csv',STATUS='OLD')
      OPEN(11,FILE='serprb_dump.csv')
C          ---- CSV header ----
      HEAD=' '
      DO 11 I=1,20
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A,I2.2)') ',spro',I
      WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A)')
     +  ',sprob,probmp,d10s,d50s,d90s,spr'
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one case per record (list-directed, 158 values) ----
20    READ(10,*,END=99) IFX,RSNI,LUFPI,DUI,DLI,SIGMI,SYGUI,SYGLI,
     A  ADSI,SUSI,SLSI,NMMODI,NRELI,DMPI,PMPI,GCDKMI,METHDI,MSPECI,
     B  HPI,SNI,TLLOWI,TLHGHI,SIGPWI,TIMEDI,RELYI
      RSN   = RSNI
      LUFP  = LUFPI
      DU    = DUI
      DL    = DLI
      SIGM  = SIGMI
      SYGU  = SYGUI
      SYGL  = SYGLI
      ADS   = ADSI
      SUS   = SUSI
      SLS   = SLSI
      NMMOD = NMMODI
      NREL  = NRELI
      DMP   = DMPI
      PMP   = PMPI
      GCDKM = GCDKMI
      METHOD= METHDI
      MSPEC = MSPECI
      SPR   = -7777.
      D10S  = -6666.
      D50S  = -6666.
      D90S  = -6666.
      DO 22 I=1,20
      HP(I)     = HPI(I)
      SN(I)     = SNI(I)
      TLLOW(I)  = TLLOWI(I)
      TLHGH(I)  = TLHGHI(I)
      SIGPOW(I) = SIGPWI(I)
      TIMED(I)  = TIMEDI(I)
      RELY(I)   = RELYI(I)
22    SPRO(I)   = -8888.
      DO 24 I=1,13
      SPROB(I)  = -8888.
24    PROBMP(I) = -8888.
      CALL SERPRB(IFX)
      CALL MPATH(IFX)
      WRITE(11,30) (SPRO(I),I=1,20),
     A SPROB(IFX),PROBMP(IFX),D10S,D50S,D90S,SPR
30    FORMAT(ES18.9E3,25(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
