      PROGRAM DUMPALLM
C     Tier-B oracle dump for ALLMODES (accumulate /ZON/ into /allMODE/).
C
C     ALLMODES has no arithmetic -- it is assignment and index selection -- so
C     what is under test is the bookkeeping: which /ZON/ slots each IFLG arm
C     takes, the HP > 0 liveness test, the NMMOD cursor, and the IFLG=1 arm's
C     reserved NMMOD+2 slot.
C
C     /allMODE/ is deliberately NOT cleared between cases: accumulation across
C     calls is the behaviour being pinned, so the case file is one long call
C     sequence with IFLG=0 resets in it.  The generator tracks NMMOD the same
C     way ALLMODES does and inserts a reset before 20 slots would be passed,
C     since the Fortran bound-checks nothing.
C
C     /ZON/ HP(1..7) is dumped as an output too: the IFLG=1 arm writes
C     HP(2) = HP(1) back into /ZON/ before copying.
      COMMON / ZON / ABPS(7), CREL(7), EFF(7), FLDST(7), GRLOS(7),
     1 HN(7), HP(7), PROB(7), RELY(7), RGAIN(7), SIGPOW(7), SN(7),
     2 SPRO(7), TGAIN(7), TIMED(7), TLOSS(7), B(7), FSLOS(7), ADV(7),
     3 OBF(7),NMODE(7),TLLOW(7),TLHGH(7)
      COMMON / allMODE /zABPS(20),zCREL(20),zFLDST(20),zHN(20),zHP(20),
     1zPROB(20),zRELY(20),zRGAIN(20),zSIGPOW(20),zSN(20),
     2zSPRO(20),zTGAIN(20),zTIMED(20),zTLOSS(20),zB(20),zFSLOS(20),
     3zgrlos(20),zadv(20),zobf(20),
     CizNMODE(20),zTLLOW(20),zTLHGH(20),zEFF(20),NREL,NMMOD
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      CHARACTER HEAD*12000, NMA(22)*7
      DIMENSION ZI(7)
      DATA NMA/'zabps','zcrel','zfldst','zhn','zhp','zprob','zrely',
     +         'zrgain','zsigpow','zsn','zspro','ztgain','ztimed',
     +         'ztloss','zb','zfslos','zgrlos','zadv','zobf','ztllow',
     +         'ztlhgh','zeff'/
      MSPEC = 0
      NMMOD = 0
C          ---- CSV header ----
      HEAD=' '
      DO 11 J=1,22
      DO 11 I=1,20
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I2.2)') ',',TRIM(NMA(J)),I
      DO 12 I=1,20
12    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A,I2.2)') ',znmode',I
      DO 13 I=1,7
13    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A,I1)') ',zonhp',I
      WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A)') ',nmmod,freq'
      OPEN(10,FILE='allmodes_cases.csv',STATUS='OLD')
      OPEN(11,FILE='allmodes_dump.csv')
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one call per record (list-directed, 163 values) ----
20    READ(10,*,END=99) IFLGI,FVALI,
     A  ABPS,CREL,FLDST,HN,HP,PROB,RELY,RGAIN,SIGPOW,SN,SPRO,TGAIN,
     B  TIMED,TLOSS,B,FSLOS,GRLOS,ADV,OBF,TLLOW,TLHGH,EFF,ZI
      DO 22 I=1,7
22    NMODE(I)=NINT(ZI(I))
      CALL ALLMODES(IFLGI,FVALI)
      WRITE(11,30) (zABPS(I),I=1,20),(zCREL(I),I=1,20),
     A (zFLDST(I),I=1,20),(zHN(I),I=1,20),(zHP(I),I=1,20),
     B (zPROB(I),I=1,20),(zRELY(I),I=1,20),(zRGAIN(I),I=1,20),
     C (zSIGPOW(I),I=1,20),(zSN(I),I=1,20),(zSPRO(I),I=1,20),
     D (zTGAIN(I),I=1,20),(zTIMED(I),I=1,20),(zTLOSS(I),I=1,20),
     E (zB(I),I=1,20),(zFSLOS(I),I=1,20),(zgrlos(I),I=1,20),
     F (zadv(I),I=1,20),(zobf(I),I=1,20),(zTLLOW(I),I=1,20),
     G (zTLHGH(I),I=1,20),(zEFF(I),I=1,20),
     H (FLOAT(izNMODE(I)),I=1,20),(HP(I),I=1,7),FLOAT(NMMOD),FREQ
30    FORMAT(ES18.9E3,468(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
