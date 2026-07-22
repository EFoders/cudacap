      PROGRAM DUMPCURMUF
C     Tier-B oracle dump for CURMUF_orig (Module-3 circuit MUF/FOT).
C     Reads curmuf_cases.csv, loads the F2D table via a real REDMAP (for
C     F2DIS), injects controlled FI/YI/HI/FS + path geometry, calls
C     CURMUF_ORIG directly (bypassing the version dispatcher), and writes
C     the /MUFS/ outputs.  IEDP=-1 -> the GETHP profile path.
      use voacapl_defs
      use crun_directory
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
      COMMON / TIME / IT, GMT, UTIME(24), GMTR, XLMT(24), ITIM, JTX
      COMMON / CON / D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON /ES /FS (3, 5), HS (5)
      COMMON /GEOG /GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5), ARTIC(5),
     A SIGPAT(5), EPSPAT(5)
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON/INFORM/INFO,IHSHR,IHLNG
      COMMON/MUFS/EMUF(24),F1MUF(24),F2MUF(24),ESMUF(24),ALLMUF(24)
     1,FOT(24),XLUF(24),HPF(24),ANGMUF(24),MODMUF,SIGL(4),SIGU(4),DELMUF
     2(4),HPMUF(4),HTMUF(4),FVMUF(4),AFMUF(4),NHOPMF(4),YFOT(4),YHPF(4)
     3 ,YMUF(4)
      DIMENSION IIA(6), IIB(6)
      CHARACTER HOMED*200, HEAD*2000, LNAME(11)*7
      INTEGER MONTHIN
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
      DATA RZ/6370./, PI/3.1415926/, VOFL/299.79246/, DCL/1.28/
      DATA GAMA/0.57721566/, PI2/6.283185307/, PIO2/1.570796326/
      DATA D2R/.01745329251/, R2D/57.295779513/
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
      DATA LNAME/'sigl','sigu','delmuf','hpmuf','htmuf','fvmuf','afmuf',
     A 'nhopmf','yfot','yhpf','ymuf'/
      NT = 20
      NPL = 1
      XNPL = NPL
      TWDIV = 1. / (2. ** NPL)
      DO 3 I=1,3
3     IELECT(I)=0
      IEDP = -1
      INFO = 0
      IT = 1
      KM = 1
      KFX = 1
      COEFF = 'CCIR'
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
C     build the wide CSV header
      HEAD='emuf,f1muf,f2muf,esmuf,allmuf,fot,hpf,angmuf,modmuf'
      DO 5 IL=1,11
      DO 5 L=1,4
5     WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I1)')
     A  ',',TRIM(LNAME(IL)),L
      OPEN(10,FILE='curmuf_cases.csv',STATUS='OLD')
      OPEN(11,FILE='curmuf_dump.csv')
      WRITE(11,'(A)') TRIM(HEAD)
10    READ(10,*,END=99) MONTHIN, SSNIN, GCDKMI, AMINDI, XLAT, XCLCK,
     A FI1,FI2,FI3, YI1,YI2,YI3, HI1,HI2,HI3, FS1,FS2,FS3
      MONTH=MONTHIN
      SSN=SSNIN
      DO 15 I=1,6
      IA(I)=IIA(I)
15    IB(I)=IIB(I)
      CALL REDMAP(SSN, MONTH)
      FI(1,1)=FI1
      FI(2,1)=FI2
      FI(3,1)=FI3
      YI(1,1)=YI1
      YI(2,1)=YI2
      YI(3,1)=YI3
      HI(1,1)=HI1
      HI(2,1)=HI2
      HI(3,1)=HI3
      FS(1,1)=FS1
      FS(2,1)=FS2
      FS(3,1)=FS3
      HS(1)=110.
      CLAT(1)=XLAT
      CLCK(1)=XCLCK
      GCDKM=GCDKMI
      GCD=GCDKMI/6370.
      AMIND=AMINDI
      CALL CURMUF_ORIG
      WRITE(11,20) EMUF(1),F1MUF(1),F2MUF(1),ESMUF(1),ALLMUF(1),FOT(1),
     A HPF(1),ANGMUF(1),FLOAT(MODMUF),
     B (SIGL(L),L=1,4),(SIGU(L),L=1,4),(DELMUF(L),L=1,4),
     C (HPMUF(L),L=1,4),(HTMUF(L),L=1,4),(FVMUF(L),L=1,4),
     D (AFMUF(L),L=1,4),(FLOAT(NHOPMF(L)),L=1,4),
     E (YFOT(L),L=1,4),(YHPF(L),L=1,4),(YMUF(L),L=1,4)
20    FORMAT(ES18.9E3,52(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
