      PROGRAM DUMPGEOTIM
C     Tier-B oracle dump for GEOTIM + CNGTIM, the UT/LMT conversions that
C     open every hour of the prediction.
C
C     CNGTIM is exercised twice over: directly, so its day-change return and
C     its in-place rewrite of TIME can both be seen, and through GEOTIM,
C     which calls it up to seven times and throws every day-change away.
C
C     The direct call is made FIRST and on its own copies, so it cannot
C     disturb the /TIME/ and /GEOG/ slots GEOTIM then fills.
      COMMON / CON / D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON / GEOG / GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5),
     1 ARTIC(5), SIGPAT(5), EPSPAT(5)
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON / TIME / IT, GMT, UTIME(24), GMTR, XLMT(24), ITIM, JTX
      DIMENSION CLONGI(5)
C     R2D as blkdat sets it -- the driver does not link blkdat, because that
C     would also DATA-initialise commons this program injects.
      DATA R2D/57.295779513/
      OPEN(10,FILE='geotim_cases.csv',STATUS='OLD')
      OPEN(11,FILE='geotim_dump.csv')
      WRITE(11,'(A)') 'ctime,cday,it,gmt,gmtr,utime,xlmt,'//
     +  'clck1,clck2,clck3,clck4,clck5'
C          ---- one case per record (list-directed, 13 values) ----
20    READ(10,*,END=99) JT,ITIMI,TLONGI,RLONGI,KMI,CLONGI,CTIMEI,
     A  CTHETI,ISWI
      ITIM  = ITIMI
      TLONG = TLONGI
      RLONG = RLONGI
      KM    = KMI
      DO 21 I=1,5
      CLONG(I) = CLONGI(I)
21    CLCK(I)  = -8888.
      IT    = -9
      GMT   = -8888.
      GMTR  = -8888.
      DO 22 I=1,24
      UTIME(I) = -8888.
22    XLMT(I)  = -8888.
C          ---- CNGTIM on its own, before GEOTIM touches anything ----
      CTIME = CTIMEI
      CDAY  = CNGTIM(CTIME,CTHETI,ISWI)
      CALL GEOTIM(JT)
C          IT is the slot GEOTIM chose; report the two hour arrays there
      IUT = IT
      IF(IUT.lt.1 .or. IUT.gt.24) IUT = 24
      WRITE(11,30) CTIME,CDAY,FLOAT(IT),GMT,GMTR,
     A  UTIME(IUT),XLMT(IUT),(CLCK(I),I=1,5)
      GO TO 20
30    FORMAT(ES18.9E3,11(',',ES18.9E3))
99    CLOSE(10)
      CLOSE(11)
      END
