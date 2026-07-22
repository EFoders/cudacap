      PROGRAM DUMPREGMOD
C     Tier-B oracle dump for REGMOD (per-mode loss budget -> /ZON/).
C
C     Every input is injected rather than produced by running the upstream
C     chain: REGMOD reads plain scalars out of /RON/, /ES/, /GEOG/, /MODES/,
C     /MUFS/, /SIGD/, /ANOIS/ and /RAYS/ and never interpolates a table, so
C     the FINDF/IFOB argument for replaying the real chain does not apply and
C     direct injection buys much tighter branch control.  The one quantized
C     input, IFOB(1,3,K), is a single integer and is injected as such.
C
C     /cantenna/ is loaded from sample.14 as in the GAIN dump, entry 1 the
C     transmitter over [TXLO,TXHI] and entry 2 the receiver over 2-30 MHz;
C     a case whose frequency falls outside the transmitter range exercises
C     both GAIN's no-match return and PWRDB's 30 dBW default.
C
C     INFO is left 0 so REGMOD's IAND(INFO,2) diagnostic writes to unit 99
C     stay off -- they compute nothing that reaches /ZON/.
      common /gh_ipfg/ ipfg
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
      COMMON /MODES /GHOP, DELMOD (6, 3), HPMOD (6, 3), HTMOD (6, 3), FV
     1MOD (6, 3), ITMOD (6, 3), AFMOD (6, 3)
      COMMON /GEOG /GYZ (5), RAT (5), GMDIP (5), CLCK (5), ABIY (5), ART
     1IC (5), SIGPAT (5), EPSPAT (5)
      COMMON / RAYS / ANG(40), IFOB(40,30,5), NANG
      COMMON/INFORM/INFO,IHSHR,IHLNG
      COMMON /CANTENNA/ NUMANTS,IATS(20),ANTTYPE(20),ANTNAME(20),
     +                  XFQS(20),XFQE(20),DESIGNFREQ(20),ANTFILE(20),
     +                  BEAMMAIN(20),OFFAZIM(20),COND(20),DIEL(20),
     +                  ARRAY(30,91,20),AEFF(30,20)
      CHARACTER ANTTYPE*10,ANTNAME*70,ANTFILE*24
      COMMON /PANTENNA/ PWRKW(20),PWRDBA(20)
      CHARACTER HEAD*3000, NMZ(20)*7, DESCR*70
      DIMENSION PARM(20), GAINS(91)
      DIMENSION SIGLI(4),SIGUI(4),YMUFI(4),YFOTI(4),YHPFI(4),
     +          HPMUFI(4),HTMUFI(4),FVMUFI(4)
      DIMENSION DELI(6),HPI(6),HTI(6),FVI(6),ITI(6),AFI(6)
      DIMENSION SIGI(5),EPSI(5)
      DATA NMZ/'abps','adv','b','eff','fldst','fslos','grlos','hn','hp',
     +         'nmode','obf','prob','rgain','sigpow','sn','tgain',
     +         'timed','tllow','tlhgh','tloss'/
      DATA RZ/6370./, R2D/57.295779513/, PIO2/1.570796326/
      DATA D2R/.01745329251/, VOFL/299.79246/
      INFO = 0
      IPFG = 0
      MSPEC = 0
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
      XFQS(2)=2.0
      XFQE(2)=30.0
      OFFAZIM(1)=0.
      OFFAZIM(2)=0.
C          ---- CSV header ----
      HEAD=' '
      DO 11 J=1,20
      DO 11 IM=1,7
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I1)') ',',TRIM(NMZ(J)),IM
      OPEN(10,FILE='regmod_cases.csv',STATUS='OLD')
      OPEN(11,FILE='regmod_dump.csv')
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one case per record (list-directed, 103 values) ----
20    READ(10,*,END=99) JMODEI,FREQI,GCDKMI,GHOPKMI,KMI,
     A  FI1,GYZI,FS1,FS2,HSI,IFOB13,
     B  ACAVI,HTLOSI,XNUZI,HNUI,XVEI,AFEI,BFEI,ASMI,DSLI,DSUI,RCNSEI,
     C  SIGLI,SIGUI,YMUFI,YFOTI,YHPFI,HPMUFI,HTMUFI,FVMUFI,
     D  DELI,HPI,HTI,FVI,ITI,AFI,SIGI,EPSI,PWRA,TXLO,TXHI
      K = JMODEI
      L = 2*K - 1
      JMODE = JMODEI
      FREQ = FREQI
      GCDKM = GCDKMI
      GCD = GCDKMI/RZ
      GHOP = GHOPKMI/RZ
      KM = KMI
      KFX = 1
      FI(1,K) = FI1
      GYZ(L) = GYZI
      FS(1,K) = FS1
      FS(2,K) = FS2
      HS(K) = HSI
      IFOB(1,3,K) = IFOB13
      ACAV = ACAVI
      HTLOSS = HTLOSI
      XNUZ = XNUZI
      HNU = HNUI
      XVE = XVEI
      AFE = AFEI
      BFE = BFEI
      ASM = ASMI
      DSL = DSLI
      DSU = DSUI
      RCNSE = RCNSEI
      DO 22 I=1,4
      SIGL(I)=SIGLI(I)
      SIGU(I)=SIGUI(I)
      YMUF(I)=YMUFI(I)
      YFOT(I)=YFOTI(I)
      YHPF(I)=YHPFI(I)
      HPMUF(I)=HPMUFI(I)
      HTMUF(I)=HTMUFI(I)
22    FVMUF(I)=FVMUFI(I)
      DO 24 IM=1,6
      DELMOD(IM,K)=DELI(IM)
      HPMOD(IM,K)=HPI(IM)
      HTMOD(IM,K)=HTI(IM)
      FVMOD(IM,K)=FVI(IM)
      ITMOD(IM,K)=ITI(IM)
24    AFMOD(IM,K)=AFI(IM)
      DO 26 I=1,5
      SIGPAT(I)=SIGI(I)
26    EPSPAT(I)=EPSI(I)
      PWRDBA(1)=PWRA
      XFQS(1)=TXLO
      XFQE(1)=TXHI
C     preset the /ZON/ outputs so any field REGMOD leaves alone on this case
C     is deterministic rather than stale from the previous one
      DO 28 IM=1,7
      ABPS(IM)=0.
      ADV(IM)=0.
      B(IM)=0.
      EFF(IM)=0.
      FLDST(IM)=0.
      FSLOS(IM)=0.
      GRLOS(IM)=0.
      HN(IM)=0.
      HP(IM)=0.
      NMODE(IM)=0
      OBF(IM)=0.
      PROB(IM)=0.
      RGAIN(IM)=0.
      SIGPOW(IM)=0.
      SN(IM)=0.
      TGAIN(IM)=0.
      TIMED(IM)=0.
      TLLOW(IM)=0.
      TLHGH(IM)=0.
28    TLOSS(IM)=0.
      CALL REGMOD
      WRITE(11,30) (ABPS(IM),IM=1,7),(ADV(IM),IM=1,7),(B(IM),IM=1,7),
     A (EFF(IM),IM=1,7),(FLDST(IM),IM=1,7),(FSLOS(IM),IM=1,7),
     B (GRLOS(IM),IM=1,7),(HN(IM),IM=1,7),(HP(IM),IM=1,7),
     C (FLOAT(NMODE(IM)),IM=1,7),(OBF(IM),IM=1,7),(PROB(IM),IM=1,7),
     D (RGAIN(IM),IM=1,7),(SIGPOW(IM),IM=1,7),(SN(IM),IM=1,7),
     E (TGAIN(IM),IM=1,7),(TIMED(IM),IM=1,7),(TLLOW(IM),IM=1,7),
     F (TLHGH(IM),IM=1,7),(TLOSS(IM),IM=1,7)
30    FORMAT(ES18.9E3,139(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
