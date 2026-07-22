      PROGRAM DUMPLAYERS
C     Tier-B oracle dump for the Module-2 layer chain: TIMVAR -> F2VAR ->
C     ESIND -> IONSET for one sample area (KM=1).  Reads layers_cases.csv
C     (month,ssn,gmt,clat,clong,glat,gmdip,gyz — angles in radians), sets
C     up state through the real REDMAP+VIRTIM, runs the chain, and writes
C     the resulting E/F1/F2/Es layer parameters.
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
      COMMON /TWO_AB/ AB(318)
      COMMON /TIME/ IT, GMT, UTIME(24), GMTR, XLMT(24), ITIM, JTX
      COMMON /CON/ D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /RON/ CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5),
     1 YI(3,5), HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5), KM,
     2 KFX, AFAC(30,5), HTR(50,3), FNSQ(50,3)
      COMMON /ES/ FS(3,5), HS(5)
      COMMON /GEOG/ GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5),
     * ARTIC(5), SIGPAT(5), EPSPAT(5)
      COMMON /MFAC/ F2M3(5), HPF2(5), ZENANG(5), ZENMAX(5), IEDP,
     * FSECV(3)
      COMMON /PSCA/ PSC(4), PSCB(4), IPSC
      COMMON /SSP/ SUN(2,12), MONTH
      COMMON /DON/ ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD, RLONG, RLONGD, BRTD, FLUX, ULAT, ULATD, ULONG, ULONGD, SSN,
     3 D90R, D50R, D10R, D90S, D50S, D10S
      COMMON /A11/ GAMMA(6)
      DIMENSION IIA(6), IIB(6)
      CHARACTER HOMED*200
      INTEGER MONTHIN
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
      DATA RZ/6370./, PI/3.1415926/, VOFL/299.79246/, DCL/1.28/
      DATA GAMA/0.57721566/, PI2/6.283185307/, PIO2/1.570796326/
      DATA D2R/.01745329251/, R2D/57.295779513/
      DATA PSC/1.,1.,1.,0.7/
      DATA SUN/-23.05,-17.31,-17.30,-7.89,-7.88,4.21,4.26,
     1 14.83, 14.84, 21.93, 21.93, 23.45, 23.15, 18.23, 18.20, 8.68,8.55,
     2 -2.86, -2.90, -14.16, -14.20, -21.68, -21.66, -23.45/
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
      COEFF='CCIR'
      KM=1
      OPEN(10,FILE='layers_cases.csv',STATUS='OLD')
      OPEN(11,FILE='layers_dump.csv')
      WRITE(11,'(A)') 'fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3,'//
     * 'fs1,fs2,fs3,hs,f2m3,hpf2,rat,abiy,clck,zenang,zenmax,kfx'
10    READ(10,*,END=99) MONTHIN, SSNIN, GMTIN, XLAT, XLONG,
     * XGLAT, XDIP, XGYZ
      MONTH=MONTHIN
      SSN=SSNIN
      DO 15 I=1,6
      IA(I)=IIA(I)
15    IB(I)=IIB(I)
      CALL REDMAP(SSN, MONTH)
      GMT=GMTIN
      CALL VIRTIM
      CLAT(1)=XLAT
      CLONG(1)=XLONG
      GLAT(1)=XGLAT
      GMDIP(1)=XDIP
      GYZ(1)=XGYZ
      CALL TIMVAR
      CALL F2VAR
      CALL ESIND
      CALL IONSET
      WRITE(11,20) FI(1,1),FI(2,1),FI(3,1),YI(1,1),YI(2,1),YI(3,1),
     * HI(1,1),HI(2,1),HI(3,1),FS(1,1),FS(2,1),FS(3,1),HS(1),
     * F2M3(1),HPF2(1),RAT(1),ABIY(1),CLCK(1),ZENANG(1),ZENMAX(1),KFX
20    FORMAT(ES18.9E3,19(',',ES18.9E3),',',I2)
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
