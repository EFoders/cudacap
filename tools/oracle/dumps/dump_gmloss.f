      PROGRAM DUMPGMLOSS
C     Tier-B oracle dump for GMLOSS + SELTXR, driven in LUFFY's own order --
C     GMLOSS (which calls SETTXR) and then SELTXR.
C
C     Both are DISPATCHERS over an original and an "Alex" variant chosen on
C     VERSN(8:8); the driver plants the vendored database/version.w32 string
C     ("16.1207W"), so both dispatchers select the ORIGINAL, which is what the
C     port implements.
C
C     The inputs are SETTXR's grid plus SELTXR's four: DELOPT, GMIN, YMIN and
C     the /INFORM/ mask.  INFO bit 4 is not cosmetic -- it makes SELTXR run
C     its whole selection twice, and the second pass inherits XHPM and YHPM
C     from the first, so the answer can differ.
C
C     TXRGML is preset to a sentinel because GMLOSS does NOT preset it: a ray
C     that fails either test keeps whatever was there.  /ZON/ slots 1 and 7
C     are dumped to show the preset loop covering the whole range.
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
      COMMON/LPATH/ GCDLNG,TXRGML(45,2),DELOPT,GMIN,YMIN,LTXRGM(2)
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
      COMMON / ZON / ZABPS(7), ZCREL(7), ZEFF(7), ZFLDST(7), ZGRLOS(7),
     1 ZHN(7), ZHP(7), ZPROB(7), ZRELY(7), ZRGAIN(7), ZSIGPO(7), ZSN(7),
     2 ZSPRO(7), ZTGAIN(7), ZTIMED(7), ZTLOSS(7), ZB(7), ZFSLOS(7),
     3 ZADV(7), ZOBF(7),NZMODE(7),ZTLLOW(7),ZTLHGH(7)
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
      CHARACTER HEAD*20000, DESCR*70, NMZ(21)*7
      DIMENSION PARM(20), GAINS(91)
      DIMENSION SIGI(5),EPSI(5),YMUFI(4),SIGLI(4),SIGUI(4)
      DIMENSION R1(12),R2(12)
      INTEGER KA(2)
      DATA NMZ/'zabps','zcrel','zeff','zfldst','zgrlos','zhn','zhp',
     +         'zprob','zrely','zrgain','zsigpo','zsn','zspro','ztgain',
     +         'ztimed','ztloss','zb','zfslos','zadv','zobf','nzmode'/
      DATA D2R/0.01745329251/, DCL/1.28/, GAMA/0.57721566/,
     A     PI/3.1415926/, PI2/6.283185307/, PIO2/1.570796326/,
     B     R2D/57.295779513/, RZ/6370./, VOFL/299.79246/
      VERSN = '16.1207W'
      MSPEC = 0
      IHSHR = 0
      IHLNG = 0
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
C          ---- SELTXR writes its diagnostics to unit 99 ----
      OPEN(99,FILE='seltxr_diag.txt')
C          ---- CSV header ----
      HEAD=' '
      DO 11 J=1,21
      DO 11 IM=1,7,6
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A1,A,I1)') ',',TRIM(NMZ(J)),IM
      DO 12 JJ=1,2
      DO 12 IA=1,45
12    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A,I1,A1,I2.2)') ',txrgml',JJ,'_',IA
      WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A)') ',ltxrgm1,ltxrgm2'
      DO 13 IK=1,2
      DO 13 IA=1,45
13    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(2(A,I1,A1,I2.2))')
     +  ',gml',IK,'_',IA,',fhp',IK,'_',IA
      OPEN(10,FILE='gmloss_cases.csv',STATUS='OLD')
      OPEN(11,FILE='gmloss_dump.csv')
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one case per record (list-directed, 87 values) ----
20    READ(10,*,END=99) ITX2,KMI,MODMUI,NHOPI,NF1,NF2,IFOB1,IFOB2,INFOI,
     A  FREQI,GCDKMI,AMINDI,HTLOSI,XNUZI,HNUI,AFEI,BFEI,XVEI,
     B  FI11,FI12,GYZ1,GYZ2,ABIY1,ABIY2,FS21,FS22,HS1,HS2,
     C  SIGI,EPSI,YMUFI,SIGLI,SIGUI,
     D  DELMUI,HPMUFI,HTMUFI,FVMUFI,AFMUFI,R1,R2,
     E  TXLO,TXHI,RXLO,RXHI,PWRA,DELOPI,GMINI,YMINI
      ITXRCP(1) = 1
      ITXRCP(2) = ITX2
      KA(1)  = 1
      KA(2)  = ITX2
      KM     = KMI
      KFX    = ITX2
      MODMUF = MODMUI
      INFO   = INFOI
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
      DELOPT = DELOPI
      GMIN   = GMINI
      YMIN   = YMINI
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
      CALL RAMPIT(1,   NF1,R1)
      CALL RAMPIT(ITX2,NF2,R2)
C          ---- sentinels ----
      DO 30 IK=1,3
      DO 30 IA=1,45
      GML(IA,IK) = -6666.
30    FHP(IA,IK) = -6666.
      DO 31 JJ=1,2
      DO 31 IA=1,45
31    TXRGML(IA,JJ) = -8888.
      DO 32 IM=1,7
      ZABPS(IM) =-9999.
      ZCREL(IM) =-9999.
      ZEFF(IM)  =-9999.
      ZFLDST(IM)=-9999.
      ZGRLOS(IM)=-9999.
      ZHN(IM)   =-9999.
      ZHP(IM)   =-9999.
      ZPROB(IM) =-9999.
      ZRELY(IM) =-9999.
      ZRGAIN(IM)=-9999.
      ZSIGPO(IM)=-9999.
      ZSN(IM)   =-9999.
      ZSPRO(IM) =-9999.
      ZTGAIN(IM)=-9999.
      ZTIMED(IM)=-9999.
      ZTLOSS(IM)=-9999.
      ZB(IM)    =-9999.
      ZFSLOS(IM)=-9999.
      ZADV(IM)  =-9999.
      ZOBF(IM)  =-9999.
32    NZMODE(IM)=-9999
      LTXRGM(1) = -3333
      LTXRGM(2) = -3333
      CALL GMLOSS
      CALL SELTXR
      WRITE(11,40)
     A (ZABPS(IM),IM=1,7,6),(ZCREL(IM),IM=1,7,6),(ZEFF(IM),IM=1,7,6),
     B (ZFLDST(IM),IM=1,7,6),(ZGRLOS(IM),IM=1,7,6),(ZHN(IM),IM=1,7,6),
     C (ZHP(IM),IM=1,7,6),(ZPROB(IM),IM=1,7,6),(ZRELY(IM),IM=1,7,6),
     D (ZRGAIN(IM),IM=1,7,6),(ZSIGPO(IM),IM=1,7,6),(ZSN(IM),IM=1,7,6),
     E (ZSPRO(IM),IM=1,7,6),(ZTGAIN(IM),IM=1,7,6),(ZTIMED(IM),IM=1,7,6),
     F (ZTLOSS(IM),IM=1,7,6),(ZB(IM),IM=1,7,6),(ZFSLOS(IM),IM=1,7,6),
     G (ZADV(IM),IM=1,7,6),(ZOBF(IM),IM=1,7,6),
     H (FLOAT(NZMODE(IM)),IM=1,7,6),
     I ((TXRGML(IA,JJ),IA=1,45),JJ=1,2),
     J FLOAT(LTXRGM(1)),FLOAT(LTXRGM(2)),
     K ((GML(IA,KA(IK)),FHP(IA,KA(IK)),IA=1,45),IK=1,2)
40    FORMAT(ES18.9E3,313(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      CLOSE(99)
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
