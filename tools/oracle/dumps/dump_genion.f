      PROGRAM DUMPGENION
C     Tier-B oracle dump for GENION (Module-4 ionogram generation).  LECDEN
C     builds the electron-density profile HTR(50,1)/FNSQ(50,1) from
C     controlled FI/YI/HI (read from genion_cases.csv), then GENION lays out
C     the 30 vertical-sounding frequencies FVERT and fills the virtual
C     (HPRIM) and true (HTRUE) reflection heights.  IEDP is read per case so
C     both branches are exercised: IEDP<0 = full Gaussian (GETHP), IEDP>=0 =
C     the fast parabolic-E / table-lookup (XLIN + BENDY/PEN or GETHP) model.
C     No coefficient chain, so the ionogram is isolated from VERSY.
      COMMON / MFAC / F2M3(5),HPF2(5),ZENANG(5),ZENMAX(5),IEDP,FSECV(3)
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON / DATR / WT(20), XT(20), NT, NPL, XNPL, TWDIV
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      CHARACTER HEAD*2000
      INTEGER IEDPIN
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
      NT = 20
      NPL = 1
      XNPL = NPL
      TWDIV = 1. / (2. ** NPL)
      KM = 1
      DO 3 I = 1,3
3     IELECT(I) = 0
C     build the wide CSV header: fvert01..30, hprim01..30, htrue01..30
      HEAD='fvert01'
      DO 5 IH=2,30
5     WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A6,I2.2)') ',fvert',IH
      DO 6 IH=1,30
6     WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A6,I2.2)') ',hprim',IH
      DO 7 IH=1,30
7     WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A6,I2.2)') ',htrue',IH
      OPEN(10,FILE='genion_cases.csv',STATUS='OLD')
      OPEN(11,FILE='genion_dump.csv')
      WRITE(11,'(A)') TRIM(HEAD)
10    READ(10,*,END=99) FI1,FI2,FI3,YI1,YI2,YI3,HI1,HI2,HI3,IEDPIN
      FI(1,1)=FI1
      FI(2,1)=FI2
      FI(3,1)=FI3
      YI(1,1)=YI1
      YI(2,1)=YI2
      YI(3,1)=YI3
      HI(1,1)=HI1
      HI(2,1)=HI2
      HI(3,1)=HI3
      IEDP=IEDPIN
      CALL LECDEN(1)
      CALL GENION(1)
      WRITE(11,20) (FVERT(IH,1),IH=1,30),(HPRIM(IH,1),IH=1,30),
     * (HTRUE(IH,1),IH=1,30)
20    FORMAT(ES18.9E3,89(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
