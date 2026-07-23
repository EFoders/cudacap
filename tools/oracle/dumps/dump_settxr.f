      PROGRAM DUMPSETTXR
C     Tier-B oracle dump for SETTXR -- the long path's transmitter and
C     receiver end tables.
C
C     SETTXR is a DISPATCHER: settxr.for defines SETTXR_ORIG and SETTXR_ALEX
C     and picks between them on VERSN(8:8).  The vendored
C     database/version.w32 reads "Version 16.1207W", so this driver plants
C     that same string and the dispatcher selects the ORIGINAL -- the one the
C     stock program runs and the one the port implements.  Planting it here
C     rather than calling SETTXR_ORIG directly means the dispatch itself is
C     part of what the dump pins.
C
C     The /REFLX/ ray tables are too wide to inject value by value, so each
C     control area's six columns are built from an injected BASE and STEP
C     over NFILL rays, with the rest left at zero (HPFLX = 0, hence dead).
C     Every ray column is a threshold input somewhere in the routine, so a
C     ramp that crosses the threshold puts rays on both sides of it.
C
C     /cantenna/ holds antennas/samples/sample.14 as entry 1 (transmitter)
C     and entry 2 (receiver) over per-case frequency ranges, so a frequency
C     outside them exercises GAIN's no-match return.
C
C     AOFX is preset to a sentinel for all 45 rays and SETTXR is expected NOT
C     to reset it -- unlike TGAINX/ANDVX/ADVX/GRLOSX, which it does reset at
C     the top of every ray.  The sentinel surviving on a dead ray is the
C     observation.
      use cversn
      COMMON/CON/D2R,DCL,GAMA,PI,PI2,PIO2,R2D,RZ,VOFL
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON/ES/FS(3,5),HS(5)
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON/GEOG/GYZ(5),RAT(5),GMDIP(5),CLCK(5),ABIY(5),ARTIC(5),SIGPAT
     A (5),EPSPAT(5)
      COMMON/LOSX/ANDVX(45,3),ADVX(45,3),AOFX(45,3),ARFX(45,3),GRLOSX(45
     A ,3),TGAINX(45,3),TLSKM(45,3),EFFlp(45),IAFTXR(3)
      COMMON/MUFS/EMUF(24),F1MUF(24),F2MUF(24),ESMUF(24)
     A ,ALLMUF(24),FOT(24),XLUF(24),HPF(24),ANGMUF(24),MODMUF,SIGL(4)
     B ,SIGU(4),DELMUF(4),HPMUF(4),HTMUF(4),FVMUF(4),AFMUF(4),NHOPMF(4)
     C ,YFOT(4),YHPF(4),YMUF(4)
      COMMON/SIGD/DSL,ASM,DSU,AGLAT,DSLF,ASMF,DSUF,ACAV,FEAV,AFE,BFE,HNU
     A ,HTLOSS,XNUZ,XVE
      COMMON/RAYS/ANG(40),IFOB(40,30,5),NANG
      COMMON/REFLX/DELFX(45,3),HPFLX(45,3),HTFLX(45,3),GDFLX(45,3),FVFLX
     A (45,3),DSKPKM(3),DELSKP(3),HPSKP(3),HTSKP(3),DMAXKM(3),FVSKP(3)
     B ,ISKP(3),IMODE(45,3),AFFLX(45,3),DELPEN(3,5),GML(45,3),FHP(45,3)
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON / allMODE /ABPS(20),CREL(20),FLDST(20),HN(20),HP(20),
     1PROB(20),RELY(20),RGAIN(20),SIGPOW(20),SN(20),
     2SPRO(20),TGAIN(20),TIMED(20),TLOSS(20),B(20),FSLOS(20),
     3grlos(20),adv(20),obf(20),
     CNMODE(20),TLLOW(20),TLHGH(20),EFF(20),NREL,NMMOD
      COMMON/INFORM/INFO,IHSHR,IHLNG
      common /cantenna/ numants,iats(20),anttype(20),antname(20),
     +                  xfqs(20),xfqe(20),designfreq(20),antfile(20),
     +                  beammain(20),offazim(20),cond(20),diel(20),
     +                  array(30,91,20),aeff(30,20)
      character anttype*10,antname*70,antfile*24
      common /pantenna/ pwrkw(20),pwrdba(20)
      CHARACTER HEAD*20000, DESCR*70, NMZ(5)*7
      DIMENSION PARM(20), GAINS(91)
      DIMENSION SIGI(5),EPSI(5),YMUFI(4),SIGLI(4),SIGUI(4)
      DIMENSION R1(12),R2(12)
C          KA maps a dump column group to a control area: 1 -> 1, 2 -> ITX2
      INTEGER KA(2)
      DATA NMZ/'andvx','advx','aofx','grlosx','tgainx'/
      DATA D2R/0.01745329251/, DCL/1.28/, GAMA/0.57721566/,
     A     PI/3.1415926/, PI2/6.283185307/, PIO2/1.570796326/,
     B     R2D/57.295779513/, RZ/6370./, VOFL/299.79246/
C          ---- the stock version string: 8th character 'W' -> _orig ----
      VERSN = '16.1207W'
      MSPEC = 0
      INFO  = 0
C          ---- load sample.14 into /cantenna/, entries 1 (TX), 2 (RX) ----
      OPEN(12,FILE='ant14.voa',STATUS='OLD')
      REWIND(12)
      READ(12,'(A)') DESCR
      READ(12,*) NP
      DO 5 I=1,NP
5     READ(12,*) PARM(I)
      DO 6 I=NP+1,20
6     PARM(I)=0.
      DO 9 IFREQ=1,30
      READ(12,'(2X,F6.1,(T10,10F7.3))') EFFA,GAINS
      AEFF(IFREQ,1)=EFFA
      AEFF(IFREQ,2)=EFFA
      DO 8 J=1,91
      ARRAY(IFREQ,J,1)=GAINS(J)+PARM(1)
8     ARRAY(IFREQ,J,2)=GAINS(J)+PARM(1)
9     CONTINUE
      CLOSE(12)
      NUMANTS=2
      IATS(1)=1
      IATS(2)=2
      OFFAZIM(1)=0.
      OFFAZIM(2)=0.
      PWRKW(1)=1.
      PWRKW(2)=1.
C          ---- CSV header ----
      HEAD=' '
      DO 11 J=1,5
      DO 11 IK=1,2
      DO 11 IA=1,45
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I1,A1,I2.2)')
     +  ',',TRIM(NMZ(J)),IK,'_',IA
      DO 12 IA=1,45
12    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A,I2.2)') ',efflp_',IA
      DO 13 IK=1,2
13    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(6(A,I1),A,I1)')
     +  ',r1delfx',IK,',r1hpflx',IK,',r1htflx',IK,',r1fvflx',IK,
     +  ',r1afflx',IK,',r1gdflx',IK,',r1imode',IK
      DO 14 IK=1,2
      DO 14 IA=1,3
14    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(2(A,I1,A1,I2.2))')
     +  ',gml',IK,'_',IA*22-21,',fhp',IK,'_',IA*22-21
      WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A)') ',prob1,prob2'
      OPEN(10,FILE='settxr_cases.csv',STATUS='OLD')
      OPEN(11,FILE='settxr_dump.csv')
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one case per record (list-directed, 83 values) ----
20    READ(10,*,END=99) ITX2,KMI,MODMUI,NHOPI,NF1,NF2,IFOB1,IFOB2,
     A  FREQI,GCDKMI,AMINDI,HTLOSI,XNUZI,HNUI,AFEI,BFEI,XVEI,
     B  FI11,FI12,GYZ1,GYZ2,ABIY1,ABIY2,FS21,FS22,HS1,HS2,
     C  SIGI,EPSI,YMUFI,SIGLI,SIGUI,
     D  DELMUI,HPMUFI,HTMUFI,FVMUFI,AFMUFI,R1,R2,
     E  TXLO,TXHI,RXLO,RXHI,PWRA
      ITXRCP(1) = 1
      ITXRCP(2) = ITX2
      KA(1)  = 1
      KA(2)  = ITX2
      KM     = KMI
      KFX    = ITX2
      MODMUF = MODMUI
      FREQ   = FREQI
      GCDKM  = GCDKMI
      GCD    = GCDKMI/RZ
      AMIND  = AMINDI
      HTLOSS = HTLOSI
      XNUZ   = XNUZI
      HNU    = HNUI
      AFE    = AFEI
      BFE    = BFEI
      XVE    = XVEI
      DO 21 J=1,4
      YMUF(J)  = YMUFI(J)
      SIGL(J)  = SIGLI(J)
      SIGU(J)  = SIGUI(J)
      DELMUF(J)= DELMUI
      HPMUF(J) = HPMUFI
      HTMUF(J) = HTMUFI
      FVMUF(J) = FVMUFI
      AFMUF(J) = AFMUFI
21    NHOPMF(J)= NHOPI
      DO 22 J=1,5
      SIGPAT(J)= SIGI(J)
22    EPSPAT(J)= EPSI(J)
      FI(1,1)  = FI11
      FI(1,ITX2)= FI12
      GYZ(1)   = GYZ1
      GYZ(ITX2)= GYZ2
      ABIY(1)  = ABIY1
      ABIY(ITX2)= ABIY2
      FS(2,1)  = FS21
      FS(2,ITX2)= FS22
      HS(1)    = HS1
      HS(ITX2) = HS2
      IFOB(1,3,1)   = IFOB1
      IFOB(1,3,ITX2)= IFOB2
      xfqs(1)=TXLO
      xfqe(1)=TXHI
      xfqs(2)=RXLO
      xfqe(2)=RXHI
      pwrdba(1)=PWRA
      pwrdba(2)=PWRA
C          ---- build the two ray tables from base + step ----
      CALL RAMPIT(1,   NF1,R1)
      CALL RAMPIT(ITX2,NF2,R2)
C          ---- sentinels.  AOFX is the one SETTXR does NOT preset ----
      DO 30 IK=1,3
      DO 30 IA=1,45
      AOFX(IA,IK)  = -8888.
      ANDVX(IA,IK) = -7777.
      ADVX(IA,IK)  = -7777.
      GRLOSX(IA,IK)= -7777.
      TGAINX(IA,IK)= -7777.
      GML(IA,IK)   = -6666.
30    FHP(IA,IK)   = -6666.
      DO 31 IA=1,45
31    EFFlp(IA) = -5555.
      PROB(1) = -4444.
      PROB(2) = -4444.
      CALL SETTXR
      WRITE(11,40) ((ANDVX(IA,KA(IK)),IA=1,45),IK=1,2),
     A ((ADVX (IA,KA(IK)),IA=1,45),IK=1,2),
     B ((AOFX (IA,KA(IK)),IA=1,45),IK=1,2),
     C ((GRLOSX(IA,KA(IK)),IA=1,45),IK=1,2),
     D ((TGAINX(IA,KA(IK)),IA=1,45),IK=1,2),
     E (EFFlp(IA),IA=1,45),
     F (DELFX(1,KA(IK)),HPFLX(1,KA(IK)),
     G   HTFLX(1,KA(IK)),FVFLX(1,KA(IK)),
     H   AFFLX(1,KA(IK)),GDFLX(1,KA(IK)),
     I   FLOAT(IMODE(1,KA(IK))),IK=1,2),
     J ((GML(IA*22-21,KA(IK)),FHP(IA*22-21,KA(IK)),
     K   IA=1,3),IK=1,2),
     L PROB(1),PROB(2)
40    FORMAT(ES18.9E3,522(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
C--------------------------------
      SUBROUTINE RAMPIT(K,NFILL,R)
C     Fill /REFLX/ column K with a linear ramp over NFILL rays; the rest
C     zeroed, which leaves HPFLX = 0 and therefore no mode.
      COMMON/REFLX/DELFX(45,3),HPFLX(45,3),HTFLX(45,3),GDFLX(45,3),FVFLX
     A (45,3),DSKPKM(3),DELSKP(3),HPSKP(3),HTSKP(3),DMAXKM(3),FVSKP(3)
     B ,ISKP(3),IMODE(45,3),AFFLX(45,3),DELPEN(3,5),GML(45,3),FHP(45,3)
      COMMON/LOSX/ANDVX(45,3),ADVX(45,3),AOFX(45,3),ARFX(45,3),GRLOSX(45
     A ,3),TGAINX(45,3),TLSKM(45,3),EFFlp(45),IAFTXR(3)
      DIMENSION R(12)
      IAFTXR(K) = NFILL
      DO 10 IA=1,45
      IF(IA.LE.NFILL)THEN
        X = IA-1
        DELFX(IA,K) = R(1)  + R(2) *X
        HPFLX(IA,K) = R(3)  + R(4) *X
        HTFLX(IA,K) = R(5)  + R(6) *X
        GDFLX(IA,K) = R(7)  + R(8) *X
        FVFLX(IA,K) = R(9)  + R(10)*X
        AFFLX(IA,K) = R(11) + R(12)*X
        IMODE(IA,K) = 1 + MOD(IA-1,4)
      ELSE
        DELFX(IA,K) = 0.
        HPFLX(IA,K) = 0.
        HTFLX(IA,K) = 0.
        GDFLX(IA,K) = 0.
        FVFLX(IA,K) = 0.
        AFFLX(IA,K) = 0.
        IMODE(IA,K) = 0
      ENDIF
10    CONTINUE
      RETURN
      END
C--------------------------------
