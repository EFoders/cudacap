      PROGRAM DUMPVERSY
C     Tier-B oracle dump driver for VERSY (Module 2 geographic series).
C     Reads versy_cases.csv (icoeff,month,ssn,gmt,clat,clong,gmdip —
C     angles in radians), establishes state through the real
C     REDMAP+VIRTIM, sets sample area 1, calls VERSY(1,6,1) and writes
C     GAMMA(1..6): Es-low/median/up, foF2 term, M(3000)F2, foE.
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
      COMMON /GEOG/ GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5),
     * ARTIC(5), SIGPAT(5), EPSPAT(5)
      COMMON /A11/ GAMMA(6)
      DIMENSION IIA(6), IIB(6)
      CHARACTER HOMED*200
      INTEGER ICOEFF, MONTH, I
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
      DATA RZ/6370./, PI/3.1415926/, VOFL/299.79246/, DCL/1.28/
      DATA GAMA/0.57721566/, PI2/6.283185307/, PIO2/1.570796326/
      DATA D2R/.01745329251/, R2D/57.295779513/
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
      OPEN(10,FILE='versy_cases.csv',STATUS='OLD')
      OPEN(11,FILE='versy_dump.csv')
      WRITE(11,'(A)') 'gamma1,gamma2,gamma3,gamma4,gamma5,gamma6'
10    READ(10,*,END=99) ICOEFF, MONTH, SSN, GMTIN, XLAT, XLONG, XDIP
      COEFF='CCIR'
      IF(ICOEFF.EQ.1) COEFF='URSI'
      DO 15 I=1,6
      IA(I)=IIA(I)
15    IB(I)=IIB(I)
      CALL REDMAP(SSN, MONTH)
      GMT=GMTIN
      CALL VIRTIM
      CLAT(1)=XLAT
      CLONG(1)=XLONG
      GMDIP(1)=XDIP
      DO 16 I=1,6
16    GAMMA(I)=0.0
      CALL VERSY(1, 6, 1)
      WRITE(11,20) (GAMMA(I),I=1,6)
20    FORMAT(ES18.9E3,5(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
