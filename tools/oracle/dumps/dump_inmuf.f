      PROGRAM DUMPINMUF
C     Tier-B oracle dump for INMUF (insert over-the-MUF / zero-distance modes,
C     then call REGMOD).
C
C     INMUF's XLIN interpolates the ionogram, so by the FINDF/IFOB rule the
C     table must come from the oracle rather than be recomputed alongside:
C     LECDEN + GENION build a real FVERT/HPRIM/HTRUE profile (AFAC by the same
C     ramp the FDIST dump uses), and those four columns are dumped as injected
C     INPUTS ahead of the outputs.  Everything else is injected from the case.
C
C     The /MUFS/ distributions are dumped too: stage 3 overwrites YMUF/SIGL/
C     YFOT/SIGU/YHPF and INMUF must put them back before returning.
C
C     INFO stays 0 so the unit-99 diagnostics are off; they reach neither
C     /MODES/ nor /ZON/.
      COMMON / MFAC / F2M3(5),HPF2(5),ZENANG(5),ZENMAX(5),IEDP,FSECV(3)
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON / DATR / WT(20), XT(20), NT, NPL, XNPL, TWDIV
      common /gh_ipfg/ ipfg
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
      CHARACTER HEAD*8000, NMZ(20)*7, NMM(6)*7, NMI(4)*6, NMU(5)*5,
     +          DESCR*70
      DIMENSION PARM(20), GAINS(91)
      DIMENSION SIGLI(4),SIGUI(4),YMUFI(4),YFOTI(4),YHPFI(4),
     +          HPMUFI(4),HTMUFI(4),FVMUFI(4)
      DIMENSION DELI(6),HPI(6),HTI(6),FVI(6),ITI(6),AFI(6)
      DIMENSION SIGI(5),EPSI(5)
      DIMENSION DELMFI(4),AFMFI(4),NHOPI(4)
      INTEGER NUMMOD
      DATA NMZ/'abps','adv','b','eff','fldst','fslos','grlos','hn','hp',
     +         'nmode','obf','prob','rgain','sigpow','sn','tgain',
     +         'timed','tllow','tlhgh','tloss'/
      DATA NMM/'delmod','hpmod','htmod','fvmod','itmod','afmod'/
      DATA NMI/'fvert','hprim','htrue','afac'/
      DATA NMU/'ymuf','sigl','yfot','sigu','yhpf'/
      DATA XT      /0.0387724175,0.1160840707,0.1926975807,0.2681521850,
     1              0.3419940908,0.4137792043,0.4830758017,0.5494671251,
     2              0.6125538897,0.6719566846,0.7273182552,0.7783056514,
     3              0.8246122308,0.8659595032,0.9020988070,0.9328128083,
     4              0.9579168192,0.9772599500,0.9907262387,0.9982377097/
      DATA WT      /0.0775059480,0.0770398182,0.0761103619,0.0747231691,
     1              0.0728865824,0.0706116474,0.0679120458,0.0648040135,
     2              0.0613062425,0.0574397691,0.0532278470,0.0486958076,
     3              0.0438709082,0.0387821680,0.0334601953,0.0279370070,
     4              0.0222458492,0.0164210584,0.0104982845,0.0045212771/
      DATA RZ/6370./, R2D/57.295779513/, PIO2/1.570796326/
      DATA D2R/.01745329251/, VOFL/299.79246/
      NT = 20
      NPL = 1
      XNPL = NPL
      TWDIV = 1. / (2. ** NPL)
      INFO = 0
      IPFG = 0
      MSPEC = 0
      DO 2 I=1,3
2     IELECT(I)=0
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
      DO 9 IFR=1,30
      READ(12,'(2X,F6.1,(T10,10F7.3))') EFFA,GAINS
      AEFF(IFR,1)=EFFA
      AEFF(IFR,2)=EFFA
      DO 8 J=1,91
      ARRAY(IFR,J,1)=GAINS(J)+PARM(1)
      ARRAY(IFR,J,2)=GAINS(J)+PARM(1)
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
      DO 11 J=1,4
      DO 11 IH=1,30
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I2.2)') ',',TRIM(NMI(J)),IH
      WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A)')
     +      ',fi1,fi2,fi3,hi1,hi2,hi3,nummod,ghop'
      DO 12 J=1,6
      DO 12 IM=1,6
12    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I1)') ',',TRIM(NMM(J)),IM
      DO 13 J=1,5
      DO 13 IL=1,4
13    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I1)') ',',TRIM(NMU(J)),IL
      DO 14 J=1,20
      DO 14 IM=1,7
14    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I1)') ',',TRIM(NMZ(J)),IM
      OPEN(10,FILE='inmuf_cases.csv',STATUS='OLD')
      OPEN(11,FILE='inmuf_dump.csv')
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one case per record (list-directed, 126 values) ----
20    READ(10,*,END=99) JMODEI,FREQI,GCDKMI,GHOPKMI,KMI,
     A  FI1,GYZI,FS1,FS2,HSI,IFOB13,
     B  ACAVI,HTLOSI,XNUZI,HNUI,XVEI,AFEI,BFEI,ASMI,DSLI,DSUI,RCNSEI,
     C  SIGLI,SIGUI,YMUFI,YFOTI,YHPFI,HPMUFI,HTMUFI,FVMUFI,
     D  DELI,HPI,HTI,FVI,ITI,AFI,SIGI,EPSI,PWRA,TXLO,TXHI,
     E  IHOPI,MODMFI,DELMFI,AFMFI,NHOPI,
     F  FI2,FI3,YI1,YI2,YI3,HI1,HI2,HI3,IEDPI
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
      FI(2,K) = FI2
      FI(3,K) = FI3
      YI(1,K) = YI1
      YI(2,K) = YI2
      YI(3,K) = YI3
      HI(1,K) = HI1
      HI(2,K) = HI2
      HI(3,K) = HI3
      IEDP = IEDPI
C     real ionogram for column K; LECDEN always writes the scratch column 1,
C     so it has to run immediately before its own GENION
      CALL LECDEN(K)
      CALL GENION(K)
      DO 21 IH=1,30
21    AFAC(IH,K)=0.1+0.03*IH
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
      MODMUF = MODMFI
      DO 22 I=1,4
      SIGL(I)=SIGLI(I)
      SIGU(I)=SIGUI(I)
      YMUF(I)=YMUFI(I)
      YFOT(I)=YFOTI(I)
      YHPF(I)=YHPFI(I)
      HPMUF(I)=HPMUFI(I)
      HTMUF(I)=HTMUFI(I)
      FVMUF(I)=FVMUFI(I)
      DELMUF(I)=DELMFI(I)
      AFMUF(I)=AFMFI(I)
22    NHOPMF(I)=NHOPI(I)
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
      NUMMOD=-1
      CALL INMUF(IHOPI,NUMMOD)
      WRITE(11,30) (FVERT(IH,K),IH=1,30),(HPRIM(IH,K),IH=1,30),
     A (HTRUE(IH,K),IH=1,30),(AFAC(IH,K),IH=1,30),
     B (FI(I,K),I=1,3),(HI(I,K),I=1,3),
     B FLOAT(NUMMOD),GHOP,
     C (DELMOD(IM,K),IM=1,6),(HPMOD(IM,K),IM=1,6),(HTMOD(IM,K),IM=1,6),
     D (FVMOD(IM,K),IM=1,6),(FLOAT(ITMOD(IM,K)),IM=1,6),
     E (AFMOD(IM,K),IM=1,6),
     F (YMUF(I),I=1,4),(SIGL(I),I=1,4),(YFOT(I),I=1,4),(SIGU(I),I=1,4),
     G (YHPF(I),I=1,4),
     H (ABPS(IM),IM=1,7),(ADV(IM),IM=1,7),(B(IM),IM=1,7),
     I (EFF(IM),IM=1,7),(FLDST(IM),IM=1,7),(FSLOS(IM),IM=1,7),
     J (GRLOS(IM),IM=1,7),(HN(IM),IM=1,7),(HP(IM),IM=1,7),
     K (FLOAT(NMODE(IM)),IM=1,7),(OBF(IM),IM=1,7),(PROB(IM),IM=1,7),
     L (RGAIN(IM),IM=1,7),(SIGPOW(IM),IM=1,7),(SN(IM),IM=1,7),
     M (TGAIN(IM),IM=1,7),(TIMED(IM),IM=1,7),(TLLOW(IM),IM=1,7),
     N (TLHGH(IM),IM=1,7),(TLOSS(IM),IM=1,7)
30    FORMAT(ES18.9E3,323(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
