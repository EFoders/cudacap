      PROGRAM DUMPGAIN
C     Tier-B oracle dump for GAIN's ground-reflection branch (ITR<0).  That
C     path reads only SIGPAT/EPSPAT from /GEOG/ and never touches the
C     /cantenna/ antenna tables (GAIN presets RAIN/EFF then jumps straight to
C     label 75), so the driver just injects the ground constants and calls
C     GAIN(-IG, DELTA, FMC, RAIN, EFF).  SIGMA<=0 exercises the "no ground
C     data" early return, and DELTA<=1e-8 the fixed 6 dB floor.
      COMMON/GEOG/GYZ(5),RAT(5),GMDIP(5),CLCK(5),ABIY(5),ARTIC(5),
     A            SIGPAT(5),EPSPAT(5)
      INTEGER IGIN
      OPEN(10,FILE='gain_cases.csv',STATUS='OLD')
      OPEN(11,FILE='gain_dump.csv')
      WRITE(11,'(A)') 'rain,eff'
10    READ(10,*,END=99) IGIN, SIGMAI, ERI, DELTAI, FMCI
      SIGPAT(IGIN)=SIGMAI
      EPSPAT(IGIN)=ERI
      CALL GAIN(-IGIN, DELTAI, FMCI, RAIN, EFF)
      WRITE(11,20) RAIN, EFF
20    FORMAT(ES18.9E3,',',ES18.9E3)
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
