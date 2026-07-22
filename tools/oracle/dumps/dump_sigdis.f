      PROGRAM DUMPSIGDIS
C     Tier-B oracle dump for SIGDIS_orig (signal-distribution + absorption).
C     A real REDMAP loads SYS/PERR (for SYSSY) and F2D; the driver injects the
C     per-area layer parameters FI/YI/HI + geomagnetic latitude GLAT + local
C     time CLCK, builds each area's ionogram with LECDEN+GENION (for the XLIN
C     D-E lookup), injects the F2/Es distribution medians YFOT/YHPF/YMUF and
C     deciles SIGL/SIGU into /MUFS/, then calls SIGDIS.  VERSN is set to end
C     'W' so the dispatcher runs SIGDIS_orig with the absorption 0.1 floor.
      use voacapl_defs
      use crun_directory
      use cversn
      COMMON /ccoeff/ COEFF
      CHARACTER COEFF*4
      COMMON /FONE/ ANEW(3), BNEW(3), ACHI(2), BCHI(2)
      COMMON /ONE/ IA(6), IB(6), IKIM(10,6), ESLCOF(5,55),
     1             ESMCOF(7,61), ESUCOF(5,55), F2COF(13,76),
     2             FM3COF(9,49), ERCOF(9,22)
      COMMON /TWO/ F2D(16,6,6), FAKP(29,16,6), FAKMAP(29,16),
     1             HMYM(29,16), FAKABP(2,6), ABMAP(2,3),
     2             DUD(5,12,5), FAM(14,12), SYS(9,16,6), PERR(9,4,6)
      COMMON / MFAC / F2M3(5),HPF2(5),ZENANG(5),ZENMAX(5),IEDP,FSECV(3)
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON / DATR / WT(20), XT(20), NT, NPL, XNPL, TWDIV
      COMMON / CON / D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON /GEOG /GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5), ARTIC(5),
     A SIGPAT(5), EPSPAT(5)
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON/MUFS/EMUF(24),F1MUF(24),F2MUF(24),ESMUF(24),ALLMUF(24)
     1,FOT(24),XLUF(24),HPF(24),ANGMUF(24),MODMUF,SIGL(4),SIGU(4),DELMUF
     2(4),HPMUF(4),HTMUF(4),FVMUF(4),AFMUF(4),NHOPMF(4),YFOT(4),YHPF(4)
     3 ,YMUF(4)
      COMMON/SIGD/DSL,ASM,DSU,AGLAT,DSLF,ASMF,DSUF,ACAV,FEAV,AFE,BFE,HNU
     A ,HTLOSS,XNUZ,XVE
      COMMON/TON/ADJ,ADS,GNOS,GOT,REL,SL,SLS,SPR,SU,SUS
     A ,XNOISE,ZNOISE,NF
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON/INFORM/INFO,IHSHR,IHLNG
      DIMENSION IIA(6), IIB(6)
      CHARACTER HOMED*200, HEAD*400
      INTEGER MONTHIN, KFXIN, JMODEIN, IEDPIN
      REAL FIA(3,3), YIA(3,3), HIA(3,3), GLA(3), CLK(3)
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
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
      DATA RZ/6370./, PI/3.1415926/, VOFL/299.79246/, DCL/1.28/
      DATA GAMA/0.57721566/, PI2/6.283185307/, PIO2/1.570796326/
      DATA D2R/.01745329251/, R2D/57.295779513/
      NT = 20
      NPL = 1
      XNPL = NPL
      TWDIV = 1. / (2. ** NPL)
      DO 3 I=1,3
3     IELECT(I)=0
      INFO = 0
      KM = 1
      COEFF = 'CCIR'
      VERSN = '00.0000W'
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
      HEAD='aglat,acav,feav,htloss,xve,xnuz,hnu,afe,bfe,asm,dsu,dsl,'//
     A 'adj,su,sl,ads,sus,sls,abiy1,abiy2,abiy3,artic1,artic2,artic3'
      OPEN(10,FILE='sigdis_cases.csv',STATUS='OLD')
      OPEN(11,FILE='sigdis_dump.csv')
      WRITE(11,'(A)') TRIM(HEAD)
10    READ(10,*,END=99) MONTHIN, SSNIN, KFXIN, JMODEIN, GCDKMI, IEDPIN,
     A ((FIA(L,IS),L=1,3),(YIA(L,IS),L=1,3),(HIA(L,IS),L=1,3),
     B   GLA(IS), CLK(IS), IS=1,3),
     C YFOT3,YFOT4,YHPF3,YHPF4,YMUF3,YMUF4,SIGL3,SIGL4,SIGU3,SIGU4
      MONTH=MONTHIN
      SSN=SSNIN
      DO 15 I=1,6
      IA(I)=IIA(I)
15    IB(I)=IIB(I)
      CALL REDMAP(SSN, MONTH)
      KFX=KFXIN
      JMODE=JMODEIN
      GCDKM=GCDKMI
      IEDP=IEDPIN
      DO 20 IS=1,3
      DO 18 L=1,3
      FI(L,IS)=FIA(L,IS)
      YI(L,IS)=YIA(L,IS)
18    HI(L,IS)=HIA(L,IS)
      GLAT(IS)=GLA(IS)
20    CLCK(IS)=CLK(IS)
C     zero then inject the F2/Es distribution medians and deciles
      DO 22 L=1,4
      YFOT(L)=0.
      YHPF(L)=0.
      YMUF(L)=0.
      SIGL(L)=0.
22    SIGU(L)=0.
      YFOT(3)=YFOT3
      YFOT(4)=YFOT4
      YHPF(3)=YHPF3
      YHPF(4)=YHPF4
      YMUF(3)=YMUF3
      YMUF(4)=YMUF4
      SIGL(3)=SIGL3
      SIGL(4)=SIGL4
      SIGU(3)=SIGU3
      SIGU(4)=SIGU4
C     build each active area's ionogram (LECDEN scratch -> GENION column)
      DO 30 K=1,KFX
      CALL LECDEN(K)
30    CALL GENION(K)
      CALL SIGDIS
      WRITE(11,40) AGLAT,ACAV,FEAV,HTLOSS,XVE,XNUZ,HNU,AFE,BFE,
     A ASM,DSU,DSL, ADJ,SU,SL,ADS,SUS,SLS,
     B ABIY(1),ABIY(2),ABIY(3),ARTIC(1),ARTIC(2),ARTIC(3)
40    FORMAT(ES18.9E3,23(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
