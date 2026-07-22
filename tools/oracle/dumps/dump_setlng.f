      PROGRAM DUMPSETLNG
C     Tier-B oracle dump for SETLNG (sample-area replication).  SETLNG does no
C     arithmetic: it copies per-area scalars, the layer parameters FS/FI/YI/HI
C     and the ionogram HPRIM/HTRUE/FVERT/AFAC/IFOB between sample-area slots,
C     picking the pattern from KFX.  Each case injects a DISTINCT exact-integer
C     sentinel into every touched cell of areas 1..5 (all < 2**24 so they are
C     exact in REAL*4), runs SETLNG, then dumps 151 witness cells that cover
C     every destination range.  A mis-sourced or out-of-range copy changes a
C     witness; a cell SETLNG must not touch keeps its own area's sentinel.
C     The Python port injects the identical pattern and reads the witnesses in
C     the same order (see tests/stage_a/test_setlng.py).
      COMMON /RAYS/ ANG(40), IFOB(40,30,5), NANG
      COMMON /ES/ FS(3,5), HS(5)
      COMMON /GEOG/ GYZ(5),RAT(5),GMDIP(5),CLCK(5),ABIY(5),ARTIC(5),
     A SIGPAT(5),EPSPAT(5)
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      REAL SCAL(13,5), LAYR(4,3,5), IONO(4,30,5), WIT(151)
      REAL BASE, BASEL, BASEI, RA, RL, RIL, RSALT
      INTEGER KFXIN, ISALT, NW, ILS(3), IAS4(4), IAS3(3), PANG(5), PIL(5)
      CHARACTER HEAD*1100
      DATA ILS /1,15,30/
      DATA IAS4 /2,3,4,5/
      DATA IAS3 /2,3,4/
      DATA PANG /1,40,1,40,20/
      DATA PIL  /1,1,30,30,15/
      KM = 1
C     header col001..col151 (the Python test owns the column meanings)
      HEAD='col001'
      DO 5 I=2,151
5     WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A4,I3.3)') ',col',I
      OPEN(10,FILE='setlng_cases.csv',STATUS='OLD')
      OPEN(11,FILE='setlng_dump.csv')
      WRITE(11,'(A)') TRIM(HEAD)
10    READ(10,*,END=99) KFXIN, ISALT
      KFX = KFXIN
      RSALT = FLOAT(ISALT)
C     inject distinct sentinels into every touched cell of areas 1..5
      DO 30 IA=1,5
      RA = FLOAT(IA)
      BASE = RA*1000.0 + RSALT
      CLAT(IA)   =  10000.0 + BASE
      CLONG(IA)  =  20000.0 + BASE
      GLAT(IA)   =  30000.0 + BASE
      RD(IA)     =  40000.0 + BASE
      GYZ(IA)    =  50000.0 + BASE
      RAT(IA)    =  60000.0 + BASE
      GMDIP(IA)  =  70000.0 + BASE
      CLCK(IA)   =  80000.0 + BASE
      ABIY(IA)   =  90000.0 + BASE
      ARTIC(IA)  = 100000.0 + BASE
      SIGPAT(IA) = 110000.0 + BASE
      EPSPAT(IA) = 120000.0 + BASE
      HS(IA)     = 130000.0 + BASE
      DO 22 L=1,3
      RL = FLOAT(L)
      BASEL = RA*10000.0 + RL*1000.0 + RSALT
      FS(L,IA) = 100000.0 + BASEL
      FI(L,IA) = 200000.0 + BASEL
      YI(L,IA) = 300000.0 + BASEL
      HI(L,IA) = 400000.0 + BASEL
22    CONTINUE
      DO 26 IL=1,30
      RIL = FLOAT(IL)
      BASEI = RA*100000.0 + RIL*1000.0 + RSALT
      HPRIM(IL,IA) = 1000000.0 + BASEI
      HTRUE(IL,IA) = 2000000.0 + BASEI
      FVERT(IL,IA) = 3000000.0 + BASEI
      AFAC(IL,IA)  = 4000000.0 + BASEI
      DO 24 IANG=1,40
24    IFOB(IANG,IL,IA) = IA*100000 + IL*1000 + IANG*10 + ISALT
26    CONTINUE
30    CONTINUE
      CALL SETLNG
C     gather the mutated arrays into flat buffers
      DO 36 IA=1,5
      SCAL(1,IA)=CLAT(IA)
      SCAL(2,IA)=CLONG(IA)
      SCAL(3,IA)=GLAT(IA)
      SCAL(4,IA)=RD(IA)
      SCAL(5,IA)=GYZ(IA)
      SCAL(6,IA)=RAT(IA)
      SCAL(7,IA)=GMDIP(IA)
      SCAL(8,IA)=CLCK(IA)
      SCAL(9,IA)=ABIY(IA)
      SCAL(10,IA)=ARTIC(IA)
      SCAL(11,IA)=SIGPAT(IA)
      SCAL(12,IA)=EPSPAT(IA)
      SCAL(13,IA)=HS(IA)
      DO 34 L=1,3
      LAYR(1,L,IA)=FS(L,IA)
      LAYR(2,L,IA)=FI(L,IA)
      LAYR(3,L,IA)=YI(L,IA)
      LAYR(4,L,IA)=HI(L,IA)
34    CONTINUE
      DO 35 IL=1,30
      IONO(1,IL,IA)=HPRIM(IL,IA)
      IONO(2,IL,IA)=HTRUE(IL,IA)
      IONO(3,IL,IA)=FVERT(IL,IA)
      IONO(4,IL,IA)=AFAC(IL,IA)
35    CONTINUE
36    CONTINUE
C     build the 151-witness row in the order the Python test reads
      NW=0
C     section 1: 13 scalars x areas {2,3,4,5}
      DO 41 J=1,13
      DO 40 M=1,4
      NW=NW+1
      WIT(NW)=SCAL(J,IAS4(M))
40    CONTINUE
41    CONTINUE
C     section 2: FS/FI/YI/HI x layers {1,2,3} x areas {2,3,4,5}
      DO 45 J=1,4
      DO 44 L=1,3
      DO 43 M=1,4
      NW=NW+1
      WIT(NW)=LAYR(J,L,IAS4(M))
43    CONTINUE
44    CONTINUE
45    CONTINUE
C     section 3: HPRIM/HTRUE/FVERT/AFAC x freqs {1,15,30} x areas {2,3,4}
      DO 49 J=1,4
      DO 48 KK=1,3
      DO 47 M=1,3
      NW=NW+1
      WIT(NW)=IONO(J,ILS(KK),IAS3(M))
47    CONTINUE
48    CONTINUE
49    CONTINUE
C     section 4: IFOB at 5 (angle,freq) witnesses x areas {2,3,4}
      DO 52 IP=1,5
      DO 51 M=1,3
      NW=NW+1
      WIT(NW)=FLOAT(IFOB(PANG(IP),PIL(IP),IAS3(M)))
51    CONTINUE
52    CONTINUE
      WRITE(11,20) (WIT(I),I=1,151)
20    FORMAT(ES18.9E3,150(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
