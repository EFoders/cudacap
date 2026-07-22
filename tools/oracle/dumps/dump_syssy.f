      PROGRAM DUMPSYSSY
C     Tier-B oracle dump for SYSSY (system-loss decile table lookup).  A real
C     REDMAP loads the SYS(9,16,6)/PERR(9,4,6) tables into /TWO/; the driver
C     then calls SYSSY(G,PT,NN,...) over a grid of geomagnetic latitude G
C     (radians, both hemispheres), reflection-area local time PT and distance
C     index NN (2 or 5).  When the case flag IZF is set, F2D(1,1,1) is forced
C     negative after REDMAP to exercise the no-data fallback branch.
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
      DIMENSION IIA(6), IIB(6)
      CHARACTER HOMED*200
      INTEGER MONTHIN, NNIN, IZF
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
      SSN = 100.0
      COEFF = 'CCIR'
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
      OPEN(10,FILE='syssy_cases.csv',STATUS='OLD')
      OPEN(11,FILE='syssy_dump.csv')
      WRITE(11,'(A)') 'fm,su,sl,fmp,sup,slp'
10    READ(10,*,END=99) MONTHIN, GIN, PTIN, NNIN, IZF
      MONTH=MONTHIN
      DO 15 I=1,6
      IA(I)=IIA(I)
15    IB(I)=IIB(I)
      CALL REDMAP(SSN, MONTH)
      IF(IZF.NE.0) F2D(1,1,1) = -1.0
      G=GIN
      PT=PTIN
      NN=NNIN
      CALL SYSSY(G,PT,NN,FM,SU,SL,FMP,SUP,SLP)
      WRITE(11,20) FM,SU,SL,FMP,SUP,SLP
20    FORMAT(ES18.9E3,5(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
