      PROGRAM DUMPSELMOD
C     Tier-B oracle dump for SELMOD (controlling sample-area selection).
C     SELMOD reads only KFX and the end areas' FI(1,*)/FI(3,*) from /RON/ and
C     writes JMODE into /FRQ/.  The driver injects those and dumps JMODE.
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      INTEGER KFXIN
      OPEN(10,FILE='selmod_cases.csv',STATUS='OLD')
      OPEN(11,FILE='selmod_dump.csv')
      WRITE(11,'(A)') 'jmode'
10    READ(10,*,END=99) KFXIN, FI11,FI12,FI13,FI31,FI33
      KFX=KFXIN
      FI(1,1)=FI11
      FI(1,2)=FI12
      FI(1,3)=FI13
      FI(3,1)=FI31
      FI(3,3)=FI33
      CALL SELMOD
      WRITE(11,'(I0)') JMODE
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
