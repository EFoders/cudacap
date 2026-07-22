      PROGRAM DUMPPROFILE
C     Tier-B oracle dump for the profile/bending primitives: LECDEN builds
C     the electron-density profile HTR(50,1)/FNSQ(50,1) from controlled
C     FI/YI/HI (read from profile_cases.csv), then GETHP is probed at four
C     fractions of the F2 critical, and BENDY/PEN are evaluated.  No
C     coefficient chain, so the primitives are isolated from VERSY.
      COMMON / MFAC / F2M3(5),HPF2(5),ZENANG(5),ZENMAX(5),IEDP,FSECV(3)
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON / DATR / WT(20), XT(20), NT, NPL, XNPL, TWDIV
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      REAL BENDY, PEN
      EXTERNAL BENDY, PEN
      CHARACTER HEAD*2000
      DATA XT      /0.0387724175,0.1160840707,0.1926975807,0.2681521850,
     1              0.3419940908,0.4137792043,0.4830758017,0.5494671251,
     2              0.6125538897,0.6719566846,0.7273182552,0.7783056514,
     3              0.8246122308,0.8659595032,0.9020988070,0.9328128083,
     4              0.9579168192,0.9772599500,0.9907262387,0.9982377097/
      DATA WT      /0.0775059480,0.0770398182,0.0761103619,0.0747231691,
     1              0.0728865824,0.0706116474,0.0679120458,0.0648040135,
     2              0.0613062425,0.0574397691,0.0532278470,0.0486958076,
     3              0.0438709082,0.0387821680,0.0334601953,0.0279370070,
     4              0.0222458492,0.0164210584,0.0104982845,0.0045212771/
      NT = 20
      NPL = 1
      XNPL = NPL
      TWDIV = 1. / (2. ** NPL)
      KM = 1
      DO 3 I = 1,3
3     IELECT(I) = 0
C     build the wide CSV header
      HEAD='htr01'
      DO 6 IH=2,50
6     WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A4,I2.2)') ',htr',IH
      DO 7 IH=1,50
7     WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A5,I2.2)') ',fnsq',IH
      WRITE(HEAD(LEN_TRIM(HEAD)+1:),'(A)')
     * ',hi2,yi2,fsecv,hp1,ht1,hp2,ht2,hp3,ht3,hp4,ht4,bendy,pen'
      OPEN(10,FILE='profile_cases.csv',STATUS='OLD')
      OPEN(11,FILE='profile_dump.csv')
      WRITE(11,'(A)') TRIM(HEAD)
10    READ(10,*,END=99) FI1,FI2,FI3,YI1,YI2,YI3,HI1,HI2,HI3
      FI(1,1)=FI1
      FI(2,1)=FI2
      FI(3,1)=FI3
      YI(1,1)=YI1
      YI(2,1)=YI2
      YI(3,1)=YI3
      HI(1,1)=HI1
      HI(2,1)=HI2
      HI(3,1)=HI3
      CALL LECDEN(1)
      FP=FI(3,1)
      CALL GETHP(0.30*FP,HP1,HT1)
      CALL GETHP(0.60*FP,HP2,HT2)
      CALL GETHP(0.90*FP,HP3,HT3)
      CALL GETHP(0.99*FP,HP4,HT4)
      BND=BENDY(3,1,0.90*FP)
      PN=PEN(1,1,0.90*FP)
      WRITE(11,20) (HTR(IH,1),IH=1,50),(FNSQ(IH,1),IH=1,50),
     * HI(2,1),YI(2,1),FSECV(1),HP1,HT1,HP2,HT2,HP3,HT3,HP4,HT4,BND,PN
20    FORMAT(ES18.9E3,112(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
