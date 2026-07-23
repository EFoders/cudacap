      PROGRAM DUMPGETTOP
C     Tier-B oracle dump for GETTOP -- the over-the-top distance, i.e. how
C     much of a long path the ray flies ABOVE the E layer.
C
C     GETTOP reads the E layer only, HI(1,jf)/FI(1,jf) for the three control
C     areas, plus /LPATH/ DELOPT and /FRQ/ FREQ.  All four are injected.
C
C     /CON/ is DATA-initialised here from blkdat's own values rather than
C     linking blkdat.for, which would forbid this program initialising any
C     of the same commons; GETTOP needs only D2R and RZ from it.  DELOPT is
C     injected rather than taken from blkdat's DATA so the grid can reach the
C     sin^2 clamp, which the real 3 degrees never does.
      COMMON/CON/D2R,DCL,GAMA,PI,PI2,PIO2,R2D,RZ,VOFL
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON/LPATH/ GCDLNG,TXRGML(45,2),DELOPT,GMIN,YMIN,LTXRGM(2)
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      DATA D2R/0.01745329251/, DCL/1.28/, GAMA/0.57721566/,
     A     PI/3.1415926/, PI2/6.283185307/, PIO2/1.570796326/,
     B     R2D/57.295779513/, RZ/6370./, VOFL/299.79246/
      OPEN(10,FILE='gettop_cases.csv',STATUS='OLD')
      OPEN(11,FILE='gettop_dump.csv')
      WRITE(11,'(A)') 'gmkm'
20    READ(10,*,END=99) DELOPI,FREQI,DIKMI,HI1,HI2,HI3,FI1,FI2,FI3
      DELOPT  = DELOPI
      FREQ    = FREQI
      HI(1,1) = HI1
      HI(1,2) = HI2
      HI(1,3) = HI3
      FI(1,1) = FI1
      FI(1,2) = FI2
      FI(1,3) = FI3
C          ---- a distinctive sentinel: GMKM is claimed to be written on
C          ---- every path through the routine, so this must never survive
      GMKM = 12345.
      CALL GETTOP(DIKMI, GMKM)
      WRITE(11,30) GMKM
30    FORMAT(ES18.9E3)
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
