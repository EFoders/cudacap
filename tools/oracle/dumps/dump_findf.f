      PROGRAM DUMPFINDF
C     Tier-B oracle dump for FINDF (area coverage / ray-set search).  The chain
C     LECDEN+GENION builds the ionogram from controlled FI/YI/HI (+IEDP), SANG
C     sets NANG from GCDKM/AMIND, and FOBBY fills the reflectrix IFOB; AFAC is
C     injected as a simple ramp (FINDF only reads it, so identical values on
C     both sides validate the AFFLX interpolation).  FINDF then fills /REFLX/
C     for area 1, which is dumped in full (45 rays x 6 arrays + IMODE) plus the
C     skip/max scalars and IAFTXR.
      COMMON / MFAC / F2M3(5),HPF2(5),ZENANG(5),ZENMAX(5),IEDP,FSECV(3)
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON / DATR / WT(20), XT(20), NT, NPL, XNPL, TWDIV
      COMMON /CON /D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON/RAYS/ANG(40),IFOB(40,30,5),NANG
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON/REFLX/DELFX(45,3),HPFLX(45,3),HTFLX(45,3),GDFLX(45,3),FVFLX
     A (45,3),DSKPKM(3),DELSKP(3),HPSKP(3),HTSKP(3),DMAXKM(3),FVSKP(3)
     B ,ISKP(3),IMODE(45,3),AFFLX(45,3),DELPEN(3,5),GML(45,3),FHP(45,3)
      COMMON/LOSX/ANDVX(45,3),ADVX(45,3),AOFX(45,3),ARFX(45,3),GRLOSX(45
     A ,3),TGAINX(45,3),TLSKM(45,3),EFFlp(45),IAFTXR(3)
      CHARACTER HEAD*4000, NM(7)*6
      INTEGER IEDPIN
      DATA NM/'delfx','gdflx','hpflx','htflx','fvflx','afflx','imode'/
      DATA ANG/0., 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0
     1,14.,16.,18.,20.,22.,24.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.
     2 ,46.,48.,50.,52.,54.,56.,60.,65.,70.,75.,80.,85.,89.99/
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
      DATA D2R/.01745329251/
      NT = 20
      NPL = 1
      XNPL = NPL
      TWDIV = 1. / (2. ** NPL)
      KM = 1
      KFX = 1
      DO 3 I=1,3
3     IELECT(I)=0
C     build the wide CSV header
      HEAD='iaftxr,dskpkm,delskp,hpskp,htskp,fvskp,iskp,dmaxkm'
      DO 7 J=1,7
      DO 7 IA=1,45
7     WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I2.2)') ',',TRIM(NM(J)),IA
      OPEN(10,FILE='findf_cases.csv',STATUS='OLD')
      OPEN(11,FILE='findf_dump.csv')
      OPEN(12,FILE='findf_ifob.csv')
      WRITE(11,'(A)') TRIM(HEAD)
10    READ(10,*,END=99) FI1,FI2,FI3,YI1,YI2,YI3,HI1,HI2,HI3,IEDPIN,
     A GCDKMI,AMINDI,FREQIN
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
      GCDKM=GCDKMI
      GCD=GCDKMI/6370.
      AMIND=AMINDI
      CALL LECDEN(1)
      CALL GENION(1)
      DO 12 IH=1,30
12    AFAC(IH,1)=0.1+0.03*IH
      CALL SANG
      CALL FOBBY(1)
      FREQ=FREQIN
C     reset the /LOSX/ ray count and the skip scalars so 0-ray cases don't
C     carry stale cross-case values (FINDF overwrites them when rays exist)
      IAFTXR(1)=0
      DELSKP(1)=0.
      HPSKP(1)=0.
      HTSKP(1)=0.
      FVSKP(1)=0.
      ISKP(1)=0
      CALL FINDF(1)
      WRITE(11,20) FLOAT(IAFTXR(1)),DSKPKM(1),DELSKP(1),HPSKP(1),
     A HTSKP(1),FVSKP(1),FLOAT(ISKP(1)),DMAXKM(1),
     B (DELFX(IA,1),IA=1,45),(GDFLX(IA,1),IA=1,45),(HPFLX(IA,1),IA=1,45),
     C (HTFLX(IA,1),IA=1,45),(FVFLX(IA,1),IA=1,45),(AFFLX(IA,1),IA=1,45),
     D (FLOAT(IMODE(IA,1)),IA=1,45)
20    FORMAT(ES18.9E3,322(',',ES18.9E3))
C     dump the reflectrix IFOB that FINDF actually searched (NANG + 40x30),
C     so the Python test injects bit-identical integers and is decoupled from
C     FOBBY's COS libm floor (validated separately)
      WRITE(12,30) NANG, ((IFOB(IA,IH,1),IH=1,30),IA=1,40)
30    FORMAT(I0,1200(',',I0))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      CLOSE(12)
      END
