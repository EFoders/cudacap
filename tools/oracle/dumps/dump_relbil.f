      PROGRAM DUMPRELBIL
C     Tier-B oracle dump for RELBIL (reliability + the combined distribution).
C
C     RELBIL is the first routine to read /allMODE/ rather than /ZON/, so the
C     injected state is a list of up to twenty candidate modes plus the noise
C     deciles and the required S/N.  Everything is injected: RELBIL
C     interpolates no upstream table.
C
C     D10R/D50R/D90R are injected AS WELL as dumped.  They live in /DON/ and
C     persist across calls, and RELBIL does not always assign them -- when the
C     only mode has HP <= 70 it assigns none of the three, and SNRLW/SNRUP/
C     SNPR then read whatever the previous frequency left.  Seeding them makes
C     that stale read observable instead of accidental.
C
C     blkdat.for is LINKED rather than transcribed, so LAYTYP comes from the
C     real source; MODE(IF) is dumped as the 1..5 index that matches it (0 if
C     unset), which tests the lookup end to end without putting strings in the
C     CSV.  Because blkdat DATA-initialises /CON/ and /ALPHA/, this driver
C     must not.
      COMMON / ALPHA / IMON(12), ITRAN(2), IRCVR(2), LBMAP(2), MODE(13),
     A MODER(13), ITLAT, ITLONG, IRLAT, IRLONG, IRLATU, IRLONGU, NYEAR
      CHARACTER IMON*3, NYEAR*5, ITRAN*10, IRCVR*10, LBMAP*10, ITLAT*1,
     A ITLONG*1, IRLAT*1, IRLONG*1, IRLATU*1, IRLONGU*1, MODE*2, MODER*2
      COMMON/ANOIS/ATNU,ATNY,CC,TM,RCNSE,DU,DL,SIGM,SXGU,SXGL,KJ,JK
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON / ION / IHRE, IHRO, IHRS, LUFP, METHOD, NOISE, NPSL
      COMMON / OUTLAB / LABEL(11), LAYTYP(5), IEAST, INORTH, ISOUTH,
     A IWEST, LABLI, LABLJ, LABLK
      CHARACTER IEAST*1, ISOUTH*1, INORTH*1, IWEST*1, LABEL*5, LAYTYP*2,
     A LABLI*5, LABLJ*5, LABLK*5
      COMMON/SIGD/DSL,ASM,DSU,AGLAT,DSLF,ASMF,DSUF,ACAV,FEAV,AFE,BFE,HNU
     A ,HTLOSS,XNUZ,XVE
      COMMON / SON / ANGLE(13), ANGLER(13), CPROB(13), DBLOS(13),
     A DBLOSL(13), DBLOSU(13), DBU(13), DELAY(13), DBW(13), NHP(13),
     B XNYNOIS(13), PROBMP(13), RELIAB(13), SNDB(13), SNPR(13),
     C SNRLW(13), SNRUP(13), SPROB(13), VHIGH(13), RNEFF(13),MDL(13)
      CHARACTER MDL*1
      COMMON /DUDL_NOIS/ DU_NOIS(13),DL_NOIS(13)
      COMMON / allMODE /ABPS(20),CREL(20),FLDST(20),HN(20),HP(20),
     1PROB(20),RELY(20),RGAIN(20),SIGPOW(20),SN(20),
     2SPRO(20),TGAIN(20),TIMED(20),TLOSS(20),B(20),FSLOS(20),
     3grlos(20),adv(20),obf(20),
     CNMODE(20),TLLOW(20),TLHGH(20),EFF(20),NREL,NMMOD
      COMMON/INFORM/INFO,IHSHR,IHLNG
      common /sncom/ snxx(13)
      common /cgains/ gaint(13),gainr(13)
      COMMON /CANTENNA/ NUMANTS,IATS(20),ANTTYPE(20),ANTNAME(20),
     +                  XFQS(20),XFQE(20),DESIGNFREQ(20),ANTFILE(20),
     +                  BEAMMAIN(20),OFFAZIM(20),COND(20),DIEL(20),
     +                  ARRAY(30,91,20),AEFF(30,20)
      CHARACTER ANTTYPE*10,ANTNAME*70,ANTFILE*24
      COMMON /PANTENNA/ PWRKW(20),PWRDBA(20)
      CHARACTER HEAD*3000
      DIMENSION HPI(20),SNI(20),TLLOWI(20),TLHGHI(20),HNI(20),SIGPWI(20),
     +          FLDSTI(20),TGAINI(20),RGAINI(20),BI(20),TIMEDI(20),
     +          TLOSSI(20),PROBI(20),EFFI(20),NMODEI(20)
      INFO = 0
      NUMANTS = 1
      IATS(1) = 1
      OPEN(10,FILE='relbil_cases.csv',STATUS='OLD')
      OPEN(11,FILE='relbil_dump.csv')
C          ---- CSV header ----
      HEAD=' '
      DO 11 I=1,20
11    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A,I2.2)') ',crel',I
      DO 12 I=1,20
12    WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A,I2.2)') ',rely',I
      WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A)')
     +  ',nrel,d10r,d50r,d90r,angle,cprob,dblos,dblosl,dblosu,dbu,'//
     +  'delay,dbw,nhp,xnynois,reliab,sndb,snpr,snrlw,snrup,vhigh,'//
     +  'rneff,du_nois,dl_nois,snxx,gaint,gainr,dslf,dsuf,modecode'
      WRITE(11,'(A)') HEAD(2:LEN_TRIM(HEAD))
C          ---- one case per record (list-directed, 314 values) ----
20    READ(10,*,END=99) IFX,FREQI,RSNI,LUFPI,DUI,DLI,RCNSEI,NMMODI,
     A  PWRA,TXLO,TXHI,D10RI,D50RI,D90RI,
     B  HPI,SNI,TLLOWI,TLHGHI,HNI,SIGPWI,FLDSTI,TGAINI,RGAINI,BI,
     C  TIMEDI,TLOSSI,PROBI,EFFI,NMODEI
      RSN   = RSNI
      LUFP  = LUFPI
      DU    = DUI
      DL    = DLI
      RCNSE = RCNSEI
      NMMOD = NMMODI
      PWRDBA(1) = PWRA
      XFQS(1) = TXLO
      XFQE(1) = TXHI
      D10R = D10RI
      D50R = D50RI
      D90R = D90RI
      DSLF = -7777.
      DSUF = -7777.
      DO 22 I=1,20
      HP(I)     = HPI(I)
      SN(I)     = SNI(I)
      TLLOW(I)  = TLLOWI(I)
      TLHGH(I)  = TLHGHI(I)
      HN(I)     = HNI(I)
      SIGPOW(I) = SIGPWI(I)
      FLDST(I)  = FLDSTI(I)
      TGAIN(I)  = TGAINI(I)
      RGAIN(I)  = RGAINI(I)
      B(I)      = BI(I)
      TIMED(I)  = TIMEDI(I)
      TLOSS(I)  = TLOSSI(I)
      PROB(I)   = PROBI(I)
      EFF(I)    = EFFI(I)
      NMODE(I)  = NMODEI(I)
      CREL(I)   = -8888.
22    RELY(I)   = -8888.
      NREL = -1
      DO 24 I=1,13
      ANGLE(I)=-8888.
      CPROB(I)=-8888.
      DBLOS(I)=-8888.
      DBLOSL(I)=-8888.
      DBLOSU(I)=-8888.
      DBU(I)=-8888.
      DELAY(I)=-8888.
      DBW(I)=-8888.
      NHP(I)=-8888
      XNYNOIS(I)=-8888.
      RELIAB(I)=-8888.
      SNDB(I)=-8888.
      SNPR(I)=-8888.
      SNRLW(I)=-8888.
      SNRUP(I)=-8888.
      VHIGH(I)=-8888.
      RNEFF(I)=-8888.
      DU_NOIS(I)=-8888.
      DL_NOIS(I)=-8888.
      SNXX(I)=-8888.
      GAINT(I)=-8888.
      GAINR(I)=-8888.
24    MODE(I)='  '
      CALL RELBIL(IFX,FREQI)
      IMC = 0
      DO 26 I=1,5
26    IF(MODE(IFX).EQ.LAYTYP(I)) IMC = I
      WRITE(11,30) (CREL(I),I=1,20),(RELY(I),I=1,20),
     A FLOAT(NREL),D10R,D50R,D90R,
     B ANGLE(IFX),CPROB(IFX),DBLOS(IFX),DBLOSL(IFX),DBLOSU(IFX),
     C DBU(IFX),DELAY(IFX),DBW(IFX),FLOAT(NHP(IFX)),XNYNOIS(IFX),
     D RELIAB(IFX),SNDB(IFX),SNPR(IFX),SNRLW(IFX),SNRUP(IFX),
     E VHIGH(IFX),RNEFF(IFX),DU_NOIS(IFX),DL_NOIS(IFX),SNXX(IFX),
     F GAINT(IFX),GAINR(IFX),DSLF,DSUF,FLOAT(IMC)
30    FORMAT(ES18.9E3,68(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
