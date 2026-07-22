      PROGRAM DUMPGENOIS
C     Tier-B oracle dump for GENOIS (combined receiver noise distribution).
C
C     A real REDMAP supplies /TWO/ (F2D for the no-data branch, FAM and DUD
C     for GENFAM); everything else GENOIS reads is a plain scalar and is
C     INJECTED.  The FINDF/IFOB replay rule does not apply -- GENOIS
C     interpolates no upstream quantized table, it only evaluates polynomials
C     in tables it reads itself -- so direct injection is the right call and
C     buys control over the branches that matter:
C
C       * F2D(1,1,1) < 0  (IZF) -- no long-term data base, fixed deciles;
C       * FREQ <= FI(3,KFX)     -- galactic noise cannot penetrate, MODE_N2=0;
C       * NOISE > 0 / = 0 / < 0 -- explicit 3 MHz level, the -164 dBW default,
C                                  and the four CCIR environment categories;
C       * DUA or DLA > 12 dB    -- Caruana's limit replaces Spaulding's sigma
C                                  (METHOD_USED 1 rather than 2).  Reached
C                                  only through the real DUD tables, i.e. at
C                                  low frequency in the southern-hemisphere
C                                  time blocks, so the case grid puts cases
C                                  there deliberately.
C
C     /cantenna/ holds sample.14 as entry 2 (the receiver) over a per-case
C     [RXLO,RXHI], so cases outside that range exercise GAIN's no-match
C     return and leave EFF at 0.
C
C     /ZON/ EFF is dumped for all SEVEN slots although GENOIS fills only six:
C     slot 7 must stay at its preset.
      use voacapl_defs
      use crun_directory
      common /c_noise/ mode_n1,mode_n2,mode_n3
      common /c_noises/ xn_atnos,xn_galactic,xn_mmn
      common /c_method_used/ method_used
      COMMON / ALPHA / IMON(12), ITRAN(2), IRCVR(2), LBMAP(2), MODE(13),
     A MODER(13), ITLAT, ITLONG, IRLAT, IRLONG, IRLATU, IRLONGU, NYEAR
      CHARACTER IMON*3, NYEAR*5, ITRAN*10, IRCVR*10, LBMAP*10, ITLAT*1,
     A ITLONG*1, IRLAT*1, IRLONG*1, IRLATU*1, IRLONGU*1, MODE*2, MODER*2
      COMMON /ccoeff/ COEFF
      CHARACTER COEFF*4
      COMMON /FONE/ ANEW(3), BNEW(3), ACHI(2), BCHI(2)
      COMMON /ONE/ IA(6), IB(6), IKIM(10,6), ESLCOF(5,55),
     1             ESMCOF(7,61), ESUCOF(5,55), F2COF(13,76),
     2             FM3COF(9,49), ERCOF(9,22)
      COMMON /TWO/ F2D(16,6,6), FAKP(29,16,6), FAKMAP(29,16),
     1             HMYM(29,16), FAKABP(2,6), ABMAP(2,3),
     2             DUD(5,12,5), FAM(14,12), SYS(9,16,6), PERR(9,4,6)
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON/ANOIS/ATNU,ATNY,CC,TM,RCNSE,DU,DL,SIGM,SYGU,SYGL,KJ,JK
      COMMON /TON /ADJ, ADS, GNOS, GOT, REL, SL, SLS
     1, SPR, SU, SUS, XNOISE, ZNOISE, NF
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON / ION / IHRE, IHRO, IHRS, LUFP, METHOD, NOISE, NPSL
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON / ZON / ABPS(7), CREL(7), EFF(7), FLDST(7), GRLOS(7),
     1 HN(7), HP(7), PROB(7), RELY(7), RGAIN(7), SIGPOW(7), SN(7),
     2 SPRO(7), TGAIN(7), TIMED(7), TLOSS(7), B(7), FSLOS(7), ADV(7),
     3 OBF(7),NMODE(7),TLLOW(7),TLHGH(7)
      COMMON /CANTENNA/ NUMANTS,IATS(20),ANTTYPE(20),ANTNAME(20),
     +                  XFQS(20),XFQE(20),DESIGNFREQ(20),ANTFILE(20),
     +                  BEAMMAIN(20),OFFAZIM(20),COND(20),DIEL(20),
     +                  ARRAY(30,91,20),AEFF(30,20)
      CHARACTER ANTTYPE*10,ANTNAME*70,ANTFILE*24
      DIMENSION IIA(6), IIB(6), PARM(20), GAINS(91)
      CHARACTER HOMED*200, DESCR*70
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
      COEFF = 'CCIR'
      NYEAR = ' 2026'
      MSPEC = 0
      KFX = 1
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
C          ---- load sample.14 into /cantenna/ as entry 2, the receiver ----
      OPEN(12,FILE='ant14.voa',STATUS='OLD')
      REWIND(12)
      READ(12,'(A)') DESCR
      READ(12,*) NP
      DO 5 I=1,NP
      READ(12,*) PARM(I)
5     CONTINUE
      DO 6 I=NP+1,20
6     PARM(I)=0.
      DO 9 IFREQ=1,30
      READ(12,'(2X,F6.1,(T10,10F7.3))') EFFA,GAINS
      AEFF(IFREQ,2)=EFFA
      DO 8 J=1,91
      ARRAY(IFREQ,J,2)=GAINS(J)+PARM(1)
8     CONTINUE
9     CONTINUE
      CLOSE(12)
      NUMANTS=2
      IATS(1)=1
      IATS(2)=2
      XFQS(1)=2.0
      XFQE(1)=30.0
      OFFAZIM(1)=0.
      OFFAZIM(2)=0.
      OPEN(10,FILE='genois_cases.csv',STATUS='OLD')
      OPEN(11,FILE='genois_dump.csv')
      WRITE(11,'(A)') 'rcnse,du,dl,sigm,sygu,sygl,gnos,xnoise,znoise,'//
     +  'eff1,eff2,eff3,eff4,eff5,eff6,eff7,'//
     +  'xn_atnos,xn_galactic,xn_mmn,method_used,mode_n2'
      MPREV = -1
C          ---- one call per record (list-directed, 14 values) ----
20    READ(10,*,END=99) MONTHIN,IZF,FREQI,NOISEI,RLATI,ATNUI,ATNYI,
     A  CCI,TMI,KJI,JKI,FI3I,RXLO,RXHI
      SSN = 100.0
      IF(MONTHIN.NE.MPREV) THEN
         MONTH=MONTHIN
         DO 15 I=1,6
         IA(I)=IIA(I)
15       IB(I)=IIB(I)
         CALL REDMAP(SSN, MONTH)
         F2DSAV = F2D(1,1,1)
         MPREV = MONTHIN
      END IF
C     restore first: an IZF case must not poison the rest of the month
      F2D(1,1,1) = F2DSAV
      IF(IZF.NE.0) F2D(1,1,1) = -1.0
      FREQ  = FREQI
      NOISE = NOISEI
      RLAT  = RLATI
      ATNU  = ATNUI
      ATNY  = ATNYI
      CC    = CCI
      TM    = TMI
      KJ    = KJI
      JK    = JKI
      FI(3,1) = FI3I
      XFQS(2) = RXLO
      XFQE(2) = RXHI
C     preset every dumped output so a field GENOIS leaves alone on this case
C     is deterministic rather than stale from the previous one
      RCNSE=0.
      DU=0.
      DL=0.
      SIGM=0.
      SYGU=0.
      SYGL=0.
      GNOS=0.
      XNOISE=0.
      ZNOISE=0.
      xn_atnos=0.
      xn_galactic=0.
      xn_mmn=0.
      method_used=0
      mode_n1=0
      mode_n2=0
      mode_n3=0
      DO 28 IM=1,7
28    EFF(IM)=0.
      CALL GENOIS
      WRITE(11,30) RCNSE,DU,DL,SIGM,SYGU,SYGL,GNOS,XNOISE,ZNOISE,
     A  (EFF(IM),IM=1,7),xn_atnos,xn_galactic,xn_mmn,
     B  FLOAT(method_used),FLOAT(mode_n2)
30    FORMAT(ES18.9E3,20(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
