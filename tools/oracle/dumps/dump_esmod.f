      PROGRAM DUMPESMOD
C     Tier-B oracle dump for ESMOD + ESREG (the sporadic-E /ZON/ slots).
C
C     The two are driven together, in the order LUFFY calls them, because
C     they write DISJOINT slots -- ESMOD 4-5, ESREG 6-7 -- and running them
C     as a pair pins that neither disturbs the other's.
C
C     ESREG has 363 lines of which 26 execute: after presetting slots 6 and 7
C     it hits an unconditional RETURN, above which the source says
C     "ERRORS IN THE CODE BELOW ... Nothing DONe here!!!".  Dumping it is
C     therefore also the empirical check on that claim: if any of the dead
C     mixed Es-F code ever ran, slot 6 would come back with computed values
C     instead of the markers.
C
C     Every input is injected (the REGMOD argument -- ESMOD reads plain
C     scalars and interpolates no upstream table).  /cantenna/ holds
C     sample.14 as entry 1 (transmitter) and entry 2 (receiver) over per-case
C     frequency ranges, so cases outside them exercise GAIN's no-match return
C     and PWRDB's 30 dBW default.
C
C     All seven /ZON/ slots are preset to 12345. (NMODE -9) before the calls,
C     so any field neither routine writes is visibly untouched rather than
C     merely zero.  Slots 4..7 are dumped.
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON/ANOIS/ATNU,ATNY,CC,TM,RCNSE,DU,DL,SIGM,SXGU,SXGL,KJ,JK
      COMMON/SIGD/DSL,ASM,DSU,AGLAT,DSLF,ASMF,DSUF,ACAV,FEAV,AFE,BFE,HNU
     A ,HTLOSS,XNUZ,XVE
      COMMON/MUFS/EMUF(24),F1MUF(24),F2MUF(24),ESMUF(24),ALLMUF(24),FOT
     A(24),XLUF(24),HPF(24),ANGMUF(24),MODMUF,SIGL(4),SIGU(4),DELMUF(4)
     B ,HPMUF(4),HTMUF(4),FVMUF(4),AFMUF(4),NHOPMF(4),YFOT(4),YHPF(4)
     C ,YMUF(4)
      COMMON /CON /D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON / ZON / ABPS(7), CREL(7), EFF(7), FLDST(7), GRLOS(7),
     1 HN(7), HP(7), PROB(7), RELY(7), RGAIN(7), SIGPOW(7), SN(7),
     2 SPRO(7), TGAIN(7), TIMED(7), TLOSS(7), B(7), FSLOS(7), ADV(7),
     3 OBF(7),NMODE(7),TLLOW(7),TLHGH(7)
      COMMON/TON/ADJ,ADS,GNOS,GOT,REL,SL,SLS
     A ,SPR,SU,SUS,XNOISE,ZNOISE,NF
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON /ES /FS (3, 5), HS (5)
      COMMON /GEOG /GYZ (5), RAT (5), GMDIP (5), CLCK (5), ABIY (5), ART
     1IC (5), SIGPAT (5), EPSPAT (5)
      COMMON / RAYS / ANG(40), IFOB(40,30,5), NANG
      COMMON /REFLX/DELFX(45,3),HPFLX(45,3),HTFLX(45,3),GDFLX(45,3),FVFLX
     A (45,3),DSKPKM(3),DELSKP(3),HPSKP(3),HTSKP(3),DMAXKM(3),FVSKP(3)
     B ,ISKP(3),IMODE(45,3),AFFLX(45,3),DELPEN(3,5),GML(45,3),FHP(45,3)
      COMMON /CANTENNA/ NUMANTS,IATS(20),ANTTYPE(20),ANTNAME(20),
     +                  XFQS(20),XFQE(20),DESIGNFREQ(20),ANTFILE(20),
     +                  BEAMMAIN(20),OFFAZIM(20),COND(20),DIEL(20),
     +                  ARRAY(30,91,20),AEFF(30,20)
      CHARACTER ANTTYPE*10,ANTNAME*70,ANTFILE*24
      COMMON /PANTENNA/ PWRKW(20),PWRDBA(20)
      CHARACTER HEAD*4000, NMZ(23)*7, DESCR*70
      DIMENSION PARM(20), GAINS(91)
      DIMENSION FS1I(5),FS2I(5),FS3I(5),HSI(5),FI1I(5),GYZI(5)
      DIMENSION SIGI(5),EPSI(5)
      DATA NMZ/'abps','adv','b','crel','eff','fldst','fslos','grlos','hn',
     +         'hp','nmode','obf','prob','rely','rgain','sigpow','sn',
     +         'spro','tgain','timed','tllow','tlhgh','tloss'/
      DATA RZ/6370./, R2D/57.295779513/, PIO2/1.570796326/
      DATA D2R/.01745329251/, VOFL/299.79246/
      MSPEC = 0
      KFX = 1
C          ---- load sample.14 into /cantenna/, entries 1 (TX) and 2 (RX) ----
      OPEN(12,FILE='ant14.voa',STATUS='OLD')
      REWIND(12)
      READ(12,'(A)') DESCR
      READ(12,*) NP
      DO 5 I=1,NP
      READ(12,*) PARM(I)
5     CONTINUE
      DO 6 I=NP+1,20
6     PARM(I)=0.
      DO 9 IFREQ=1,30
      READ(12,'(2X,F6.1,(T10,10F7.3))') EFFA,GAINS
      AEFF(IFREQ,1)=EFFA
      AEFF(IFREQ,2)=EFFA
      DO 8 J=1,91
      ARRAY(IFREQ,J,1)=GAINS(J)+PARM(1)
      ARRAY(IFREQ,J,2)=GAINS(J)+PARM(1)
8     CONTINUE
9     CONTINUE
      CLOSE(12)
      NUMANTS=2
      IATS(1)=1
      IATS(2)=2
      OFFAZIM(1)=0.
      OFFAZIM(2)=0.
C          ---- CSV header: 23 fields x slots 4..7 ----
      HEAD=' '
      DO 11 J=1,23
      DO 11 IM=4,7
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I1)') ',',TRIM(NMZ(J)),IM
      OPEN(10,FILE='esmod_cases.csv',STATUS='OLD')
      OPEN(11,FILE='esmod_dump.csv')
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one case per record (list-directed, 63 values) ----
20    READ(10,*,END=99) JMODEI,FREQI,GCDKMI,AMINDI,KMI,IFOB13,
     A  ACAVI,AFEI,BFEI,ASMI,DSLI,DSUI,RCNSEI,
     B  SIGL4,SIGU4,YMUF4,YFOT4,YHPF4,
     C  FS1I,FS2I,FS3I,HSI,FI1I,GYZI,SIGI,EPSI,
     D  PWRA,TXLO,TXHI,RXLO,RXHI
      JMODE = JMODEI
      FREQ  = FREQI
      GCDKM = GCDKMI
      GCD   = GCDKMI/RZ
      AMIND = AMINDI
      KM    = KMI
      IFOB(1,3,JMODE) = IFOB13
      ACAV  = ACAVI
      AFE   = AFEI
      BFE   = BFEI
      ASM   = ASMI
      DSL   = DSLI
      DSU   = DSUI
      RCNSE = RCNSEI
      SIGL(4) = SIGL4
      SIGU(4) = SIGU4
      YMUF(4) = YMUF4
      YFOT(4) = YFOT4
      YHPF(4) = YHPF4
      DO 22 I=1,5
      FS(1,I) = FS1I(I)
      FS(2,I) = FS2I(I)
      FS(3,I) = FS3I(I)
      HS(I)   = HSI(I)
      FI(1,I) = FI1I(I)
      GYZ(I)  = GYZI(I)
      SIGPAT(I) = SIGI(I)
22    EPSPAT(I) = EPSI(I)
      PWRDBA(1) = PWRA
      XFQS(1) = TXLO
      XFQE(1) = TXHI
      XFQS(2) = RXLO
      XFQE(2) = RXHI
C     a sentinel, not zero: a field neither routine writes stays visible
      DO 28 IM=1,7
      ABPS(IM)=12345.
      ADV(IM)=12345.
      B(IM)=12345.
      CREL(IM)=12345.
      EFF(IM)=12345.
      FLDST(IM)=12345.
      FSLOS(IM)=12345.
      GRLOS(IM)=12345.
      HN(IM)=12345.
      HP(IM)=12345.
      NMODE(IM)=-9
      OBF(IM)=12345.
      PROB(IM)=12345.
      RELY(IM)=12345.
      RGAIN(IM)=12345.
      SIGPOW(IM)=12345.
      SN(IM)=12345.
      SPRO(IM)=12345.
      TGAIN(IM)=12345.
      TIMED(IM)=12345.
      TLLOW(IM)=12345.
      TLHGH(IM)=12345.
28    TLOSS(IM)=12345.
      CALL ESMOD
      CALL ESREG
      WRITE(11,30) (ABPS(IM),IM=4,7),(ADV(IM),IM=4,7),(B(IM),IM=4,7),
     A (CREL(IM),IM=4,7),(EFF(IM),IM=4,7),(FLDST(IM),IM=4,7),
     B (FSLOS(IM),IM=4,7),(GRLOS(IM),IM=4,7),(HN(IM),IM=4,7),
     C (HP(IM),IM=4,7),(FLOAT(NMODE(IM)),IM=4,7),(OBF(IM),IM=4,7),
     D (PROB(IM),IM=4,7),(RELY(IM),IM=4,7),(RGAIN(IM),IM=4,7),
     E (SIGPOW(IM),IM=4,7),(SN(IM),IM=4,7),(SPRO(IM),IM=4,7),
     F (TGAIN(IM),IM=4,7),(TIMED(IM),IM=4,7),(TLLOW(IM),IM=4,7),
     G (TLHGH(IM),IM=4,7),(TLOSS(IM),IM=4,7)
30    FORMAT(ES18.9E3,91(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
