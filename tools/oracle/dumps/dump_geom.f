      PROGRAM DUMPGEOM
C     Tier-B oracle dump driver for GEOM (links real MAGVAR/MAGFIN; the
C     NOISY land/sea lookup is stubbed below with a rule the Python test
C     mirrors exactly). Reads geom_cases.csv
C     (pole_lat,pole_lon,tlatd,tlongd,rlatd,rlongd,npsl — degrees),
C     zeroes outputs, calls GEOM, writes the /DON/ scalars and the
C     per-sample-area arrays.
      COMMON /Cnorth_pole/ G_MAGNETIC_LAT, G_MAGNETIC_LON
      COMMON /CON/ D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /DON/ ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR,
     1 GCD, GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR,
     2 RLAT, RLATD, RLONG, RLONGD, BRTD, FLUX, ULAT, ULATD, ULONG,
     3 ULONGD, SSN, D90R, D50R, D10R, D90S, D50S, D10S
      COMMON /ION/ IHRE, IHRO, IHRS, LUFP, METHOD, NOISE, NPSL
      COMMON /RON/ CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5),
     1 YI(3,5), HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5), KM,
     2 KFX, AFAC(30,5), HTR(50,3), FNSQ(50,3)
      COMMON /GEOG/ GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5),
     * ARTIC(5), SIGPAT(5), EPSPAT(5)
      INTEGER I
      DATA RZ/6370./, PI/3.1415926/, VOFL/299.79246/, DCL/1.28/
      DATA GAMA/0.57721566/, PI2/6.283185307/, PIO2/1.570796326/
      DATA D2R/.01745329251/, R2D/57.295779513/
      OPEN(10,FILE='geom_cases.csv',STATUS='OLD')
      OPEN(11,FILE='geom_dump.csv')
      WRITE(11,'(7A)') 'km,dlong,gcd,gcdkm,btr,btrd,brtd,alatd,rlatd,',
     *'rd1,rd2,rd3,rd4,rd5,clat1,clat2,clat3,clat4,clat5,',
     *'clong1,clong2,clong3,clong4,clong5,',
     *'glat1,glat2,glat3,glat4,glat5,gyz1,gyz2,gyz3,gyz4,gyz5,',
     *'gmdip1,gmdip2,gmdip3,gmdip4,gmdip5,',
     *'sigpat1,sigpat2,sigpat3,sigpat4,sigpat5,',
     *'epspat1,epspat2,epspat3,epspat4,epspat5'
10    READ(10,*,END=99) GMLA, GMLO, TLATD, TLONGD, RLATD, RLONGD, NPSL
      G_MAGNETIC_LAT = GMLA
      G_MAGNETIC_LON = GMLO
      DLONG=0.0
      GCD=0.0
      GCDKM=0.0
      BTR=0.0
      BTRD=0.0
      BRTD=0.0
      ALATD=0.0
      KM=0
      DO 15 I=1,5
      RD(I)=0.0
      CLAT(I)=0.0
      CLONG(I)=0.0
      GLAT(I)=0.0
      GYZ(I)=0.0
      GMDIP(I)=0.0
      SIGPAT(I)=0.0
      EPSPAT(I)=0.0
15    CONTINUE
      CALL GEOM
      WRITE(11,20) KM, DLONG, GCD, GCDKM, BTR, BTRD, BRTD, ALATD,
     * RLATD, (RD(I),I=1,5), (CLAT(I),I=1,5), (CLONG(I),I=1,5),
     * (GLAT(I),I=1,5), (GYZ(I),I=1,5), (GMDIP(I),I=1,5),
     * (SIGPAT(I),I=1,5), (EPSPAT(I),I=1,5)
20    FORMAT(I1,48(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
C--------------------------------------------------------------------
      SUBROUTINE NOISY(J, XLAT, XLON, WLD)
C     Land/sea stub for the dump seam: NOISY mode 7 in the real program
C     reads the land-mass map (module 7 territory). Here: the east
C     hemisphere (east longitude >= 180 deg) is land (WLD=+1), the west
C     is sea (WLD=-1). tests/stage_a/test_geom.py injects the SAME rule,
C     so both the land and sea ground-constant branches are exercised
C     without the map data.
      INTEGER J
      WLD = -1.0
      IF (XLON .GE. 180.0) WLD = 1.0
      RETURN
      END
