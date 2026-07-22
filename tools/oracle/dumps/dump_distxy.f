      PROGRAM DUMPDISTXY
C     Tier-B oracle dump driver for DISTXY (wraps DAZEL).
C     Reads distxy_cases.csv (idistance,ndistance,tlat,tlon,rlat,rlon,
C     npsl — angles in radians), zeroes COMMON/AZEL/ and the outputs,
C     calls DISTXY, writes rlatd/rlongd plus the AZEL state.
      COMMON /AZEL/ ZTLAT,ZTLON,ZTHT,ZRLAT,ZRLON,ZRHT,ZTAZ,ZRAZ,
     * ZTELV,ZRELV,ZD,ZDGC,ZTAKOF,ZRAKOF
      COMMON /CON/ D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      INTEGER ID, ND, NPSL
      DATA RZ/6370./, PI/3.1415926/, VOFL/299.79246/, DCL/1.28/
      DATA GAMA/0.57721566/, PI2/6.283185307/, PIO2/1.570796326/
      DATA D2R/.01745329251/, R2D/57.295779513/
      OPEN(10,FILE='distxy_cases.csv',STATUS='OLD')
      OPEN(11,FILE='distxy_dump.csv')
      WRITE(11,'(A,A)') 'rlatd,rlongd,ztlat,ztlon,ztht,zrlat,zrlon,',
     *'zrht,ztaz,zraz,ztelv,zrelv,zd,zdgc,ztakof,zrakof'
10    READ(10,*,END=99) ID,ND,TLATC,TLONC,RLATC,RLONC,NPSL
      ZTLAT=0.0
      ZTLON=0.0
      ZTHT=0.0
      ZRLAT=0.0
      ZRLON=0.0
      ZRHT=0.0
      ZTAZ=0.0
      ZRAZ=0.0
      ZTELV=0.0
      ZRELV=0.0
      ZD=0.0
      ZDGC=0.0
      ZTAKOF=0.0
      ZRAKOF=0.0
      RLATD=0.0
      RLONGD=0.0
      CALL DISTXY(ID,ND,TLATC,TLONC,RLATC,RLONC,NPSL,RLATD,RLONGD)
      WRITE(11,20) RLATD,RLONGD,ZTLAT,ZTLON,ZTHT,ZRLAT,ZRLON,ZRHT,
     *             ZTAZ,ZRAZ,ZTELV,ZRELV,ZD,ZDGC,ZTAKOF,ZRAKOF
20    FORMAT(ES18.9E3,15(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
