      PROGRAM DUMPMAGVAR
C     Tier-B oracle dump driver for MAGVAR/MAGFIN (PLAN.md section 3B).
C     Reads magvar_cases.csv (cenlat_rad,cenlg_rad), calls MAGVAR with
C     sample index 1, writes gyrofrequency, Rawer dip and east longitude.
      COMMON /CON/ D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /GEOG/ GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5),
     * ARTIC(5), SIGPAT(5), EPSPAT(5)
      DATA RZ/6370./, PI/3.1415926/, VOFL/299.79246/, DCL/1.28/
      DATA GAMA/0.57721566/, PI2/6.283185307/, PIO2/1.570796326/
      DATA D2R/.01745329251/, R2D/57.295779513/
      OPEN(10,FILE='magvar_cases.csv',STATUS='OLD')
      OPEN(11,FILE='magvar_dump.csv')
      WRITE(11,'(A)') 'gyz,gmdip,clg'
10    READ(10,*,END=99) CENLAT, CENLG
      GYZ(1)=0.0
      GMDIP(1)=0.0
      CLG=0.0
      CALL MAGVAR(CENLAT, CENLG, 1, CLG)
      WRITE(11,20) GYZ(1), GMDIP(1), CLG
20    FORMAT(ES18.9E3,2(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
