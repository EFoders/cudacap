      PROGRAM DUMPGANT
C     Tier-B oracle dump for GAIN's antenna branch (ITR>=0, point-to-point).
C
C     GAIN reads its table straight out of /cantenna/, which is normally filled
C     by ANTCALC from a .VOA card deck -- card parsing that is out of scope.
C     So the driver injects the table itself, following the same rule the FINDF
C     dump uses for IFOB: feed the routine under test a table the oracle
C     produced, never one recomputed alongside it.  The table is sample.14 read
C     through readant.for's own FORMAT, already validated bit-exact by the
C     READANT dump, so the Python side can load the same file and agree.
C
C     Three entries exercise the DO-10 selection scan:
C        1  transmitter,  2-15 MHz   table as read
C        2  receiver,     2-30 MHz   table as read
C        3  transmitter, 15-30 MHz   table + 3 dB, EFF + 0.5
C     so a transmitter above 15 MHz must skip entry 1, pass over the receiver
C     entry, and land on 3.  Cases below 2 MHz match nothing and return 0.
      COMMON /CON /D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /CANTENNA/ NUMANTS,IATS(20),ANTTYPE(20),ANTNAME(20),
     +                  XFQS(20),XFQE(20),DESIGNFREQ(20),ANTFILE(20),
     +                  BEAMMAIN(20),OFFAZIM(20),COND(20),DIEL(20),
     +                  ARRAY(30,91,20),AEFF(30,20)
      CHARACTER ANTTYPE*10,ANTNAME*70,ANTFILE*24
      CHARACTER*70 DESCR
      DIMENSION PARM(20), GAINS(91)
      DATA RZ/6370./, R2D/57.295779513/, PIO2/1.570796326/
      DATA D2R/.01745329251/
C          ---- load sample.14 exactly as readant.for does ----
      OPEN(10,FILE='ant14.voa',STATUS='OLD')
      REWIND(10)
      READ(10,'(A)') DESCR
      READ(10,*) N
      DO 10 I=1,N
      READ(10,*) PARM(I)
10    CONTINUE
      DO 11 I=N+1,20
11    PARM(I)=0.
      DO 14 IFREQ=1,30
      READ(10,'(2X,F6.1,(T10,10F7.3))') EFF,GAINS
      DO 12 IAT=1,3
      AEFF(IFREQ,IAT)=EFF
12    CONTINUE
      AEFF(IFREQ,3)=AEFF(IFREQ,3)+0.5
      DO 13 J=1,91
C          ANTCALC adds PARM(1), the file's max gain, to every tabulated value
      ARRAY(IFREQ,J,1)=GAINS(J)+PARM(1)
      ARRAY(IFREQ,J,2)=GAINS(J)+PARM(1)
      ARRAY(IFREQ,J,3)=GAINS(J)+PARM(1)+3.0
13    CONTINUE
14    CONTINUE
      CLOSE(10)
C          ---- /cantenna/ bookkeeping ----
      NUMANTS=3
      IATS(1)=1
      XFQS(1)=2.0
      XFQE(1)=15.0
      IATS(2)=2
      XFQS(2)=2.0
      XFQE(2)=30.0
      IATS(3)=1
      XFQS(3)=15.0
      XFQE(3)=30.0
      DO 15 IAT=1,3
15    OFFAZIM(IAT)=0.
C          ---- sweep ----
      OPEN(9,FILE='gainant_cases.csv',STATUS='OLD')
      OPEN(11,FILE='gainant_dump.csv')
      WRITE(11,'(A)') 'rain,eff'
20    READ(9,*,END=99) ITRIN, DELTAI, FMCI
      CALL GAIN(ITRIN, DELTAI, FMCI, RAIN, EFF)
      WRITE(11,30) RAIN, EFF
30    FORMAT(ES18.9E3,',',ES18.9E3)
      GO TO 20
99    CLOSE(9)
      CLOSE(11)
      END
