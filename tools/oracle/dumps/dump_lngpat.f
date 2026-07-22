      PROGRAM DUMPLNGPAT
C     Tier-B oracle dump for LNGPAT -- the long path's single mode.
C
C     LNGPAT reads one ray at each end (LTXRGM(1) at control area 1 and
C     LTXRGM(2) at area ITXRCP(2)) and averages the two into one mode.  Only
C     those two slots of the (45,3) tables are ever touched, so the driver
C     injects exactly two values per table rather than whole arrays.  They
C     cannot collide: LUFFY sets ITXRCP(2) from KFX and rewrites 1 to 2, so
C     the receiver area is never 1.
C
C     LNGPAT calls CONVH, GETTOP, TABS, BABS, PRBMUF and PWRDB, all linked
C     real.  PWRDB scans /cantenna/, so one transmitter entry spanning the
C     whole band is planted; NUMANTS = 0 reaches its 30 dBW fallback.
C
C     /CON/ is DATA-initialised here from blkdat's own values rather than
C     linking blkdat.for, which would forbid this program initialising any of
C     the same commons.  /INFORM/ is left zeroed so LNGPAT's diagnostic
C     WRITEs to unit 99 never fire.
      COMMON/CON/D2R,DCL,GAMA,PI,PI2,PIO2,R2D,RZ,VOFL
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON / GEOG / GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5),
     1 ARTIC(5), SIGPAT(5), EPSPAT(5)
      COMMON/LPATH/ GCDLNG,TXRGML(45,2),DELOPT,GMIN,YMIN,LTXRGM(2)
      COMMON/LOSX/ANDVX(45,3),ADVX(45,3),AOFX(45,3),ARFX(45,3),GRLOSX(45
     A ,3),TGAINX(45,3),TLSKM(45,3),EFFlp(45),IAFTXR(3)
      COMMON/MUFS/EMUF(24),F1MUF(24),F2MUF(24),ESMUF(24),ALLMUF(24),FOT
     A (24),XLUF(24),HPF(24),ANGMUF(24),MODMUF,SIGL(4),SIGU(4),DELMUF(4)
     B ,HPMUF(4),HTMUF(4),FVMUF(4),AFMUF(4),NHOPMF(4),YFOT(4),YHPF(4)
     C ,YMUF(4)
      COMMON/REFLX/DELFX(45,3),HPFLX(45,3),HTFLX(45,3),GDFLX(45,3),FVFLX
     A (45,3),DSKPKM(3),DELSKP(3),HPSKP(3),HTSKP(3),DMAXKM(3),FVSKP(3)
     B ,ISKP(3),IMODE(45,3),AFFLX(45,3),DELPEN(3,5),GML(45,3),FHP(45,3)
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON/SIGD/DSL,ASM,DSU,AGLAT,DSLF,ASMF,DSUF,ACAV,FEAV,AFE,BFE,HNU
     A ,HTLOSS,XNUZ,XVE
      COMMON/ANOIS/ATNU,ATNY,CC,TM,RCNSE,DU,DL,SIGM,SYGU,SYGL,KJ,JK
      COMMON / allMODE /ABPS(20),CREL(20),FLDST(20),HN(20),HP(20),
     1PROB(20),RELY(20),RGAIN(20),SIGPOW(20),SN(20),
     2SPRO(20),TGAIN(20),TIMED(20),TLOSS(20),B(20),FSLOS(20),
     3grlos(20),adv(20),obf(20),
     CNMODE(20),TLLOW(20),TLHGH(20),EFF(20),NREL,NMMOD
      common /cantenna/ numants,iats(20),anttype(20),antname(20),
     +                  xfqs(20),xfqe(20),designfreq(20),antfile(20),
     +                  beammain(20),offazim(20),cond(20),diel(20),
     +                  array(30,91,20),aeff(30,20)
      character anttype*10,antname*70,antfile*24
      common /pantenna/ pwrkw(20),pwrdba(20)
      DIMENSION YMUFI(4),YFOTI(4),YHPFI(4),FVMUFI(4),SIGLI(4),SIGUI(4)
      DATA D2R/0.01745329251/, DCL/1.28/, GAMA/0.57721566/,
     A     PI/3.1415926/, PI2/6.283185307/, PIO2/1.570796326/,
     B     R2D/57.295779513/, RZ/6370./, VOFL/299.79246/
      OPEN(10,FILE='lngpat_cases.csv',STATUS='OLD')
      OPEN(11,FILE='lngpat_dump.csv')
      WRITE(11,'(A)') 'nmmod,grlos1,hn1,hp1,rgain1,tgain1,eff1,timed1,'//
     +  'tloss1,b1,b2,nmode1,nmode2,fslos1,adv1,abps1,obf1,sigpow1,'//
     +  'fldst1,sn1,tllow1,tlhgh1,prob1,dslf,dsuf'
C          ---- one transmitter entry spanning the whole band ----
      iats(1)  = 1
      xfqs(1)  = 0.
      xfqe(1)  = 100.
20    READ(10,*,END=99) L1,L2,ITX2,IMOD1,IMOD2,NUMANI,
     A  FREQI,GCDI,DELF1,DELF2,HPFL1,HPFL2,GDFL1,GDFL2,
     B  ANDV1,ANDV2,ADVX1,ADVX2,AOFX1,AOFX2,GRLS1,GRLS2,TGNX1,TGNX2,
     C  EFFI,ASMI,DSLI,DSUI,RCNSI,ACAVI,GYZ3I,PWRI,PROB1,PROB2,DELOPI,
     D  HI1,HI2,HI3,FI1,FI2,FI3,YMUFI,YFOTI,YHPFI,FVMUFI,SIGLI,SIGUI
      LTXRGM(1) = L1
      LTXRGM(2) = L2
      ITXRCP(1) = 1
      ITXRCP(2) = ITX2
      FREQ   = FREQI
      GCD    = GCDI
      ASM    = ASMI
      DSL    = DSLI
      DSU    = DSUI
      RCNSE  = RCNSI
      ACAV   = ACAVI
      GYZ(3) = GYZ3I
      DELOPT = DELOPI
      numants  = NUMANI
      pwrdba(1)= PWRI
      HI(1,1)= HI1
      HI(1,2)= HI2
      HI(1,3)= HI3
      FI(1,1)= FI1
      FI(1,2)= FI2
      FI(1,3)= FI3
      DO 21 J=1,4
      YMUF(J) = YMUFI(J)
      YFOT(J) = YFOTI(J)
      YHPF(J) = YHPFI(J)
      FVMUF(J)= FVMUFI(J)
      SIGL(J) = SIGLI(J)
21    SIGU(J) = SIGUI(J)
C          ---- the two chosen rays.  L1/L2 are only in range when the
C          ---- routine is going to use them; the early-return cases pass
C          ---- non-positive indices, which must never be dereferenced.
      IF(L1.GE.1 .AND. L1.LE.45)THEN
        DELFX(L1,1) = DELF1
        HPFLX(L1,1) = HPFL1
        GDFLX(L1,1) = GDFL1
        IMODE(L1,1) = IMOD1
        ANDVX(L1,1) = ANDV1
        ADVX (L1,1) = ADVX1
        AOFX (L1,1) = AOFX1
        GRLOSX(L1,1)= GRLS1
        TGAINX(L1,1)= TGNX1
      ENDIF
      IF(L2.GE.1 .AND. L2.LE.45)THEN
        DELFX(L2,ITX2) = DELF2
        HPFLX(L2,ITX2) = HPFL2
        GDFLX(L2,ITX2) = GDFL2
        IMODE(L2,ITX2) = IMOD2
        ANDVX(L2,ITX2) = ANDV2
        ADVX (L2,ITX2) = ADVX2
        AOFX (L2,ITX2) = AOFX2
        GRLOSX(L2,ITX2)= GRLS2
        TGAINX(L2,ITX2)= TGNX2
        EFFlp(L2)      = EFFI
      ENDIF
C          ---- distinctive sentinels: the early return must leave every
C          ---- one of these intact, so none of them may be 0
      NMMOD = -4444
      DSLF  = -3333
      DSUF  = -2222
      PROB(1) = PROB1
      PROB(2) = PROB2
      grlos(1)= -8888.
      HN(1)   = -8888.
      HP(1)   = -8888.
      RGAIN(1)= -8888.
      TGAIN(1)= -8888.
      EFF(1)  = -8888.
      TIMED(1)= -8888.
      TLOSS(1)= -8888.
      B(1)    = -8888.
      B(2)    = -8888.
      NMODE(1)= -8888
      NMODE(2)= -8888
      FSLOS(1)= -8888.
      adv(1)  = -8888.
      ABPS(1) = -8888.
      obf(1)  = -8888.
      SIGPOW(1)= -8888.
      FLDST(1)= -8888.
      SN(1)   = -8888.
      TLLOW(1)= -8888.
      TLHGH(1)= -8888.
      CALL LNGPAT
      WRITE(11,30) FLOAT(NMMOD),grlos(1),HN(1),HP(1),RGAIN(1),TGAIN(1),
     A  EFF(1),TIMED(1),TLOSS(1),B(1),B(2),FLOAT(NMODE(1)),
     B  FLOAT(NMODE(2)),FSLOS(1),adv(1),ABPS(1),obf(1),SIGPOW(1),
     C  FLDST(1),SN(1),TLLOW(1),TLHGH(1),PROB(1),DSLF,DSUF
30    FORMAT(ES18.9E3,24(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
