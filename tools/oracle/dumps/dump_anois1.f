      PROGRAM DUMPANOIS1
C     Tier-B oracle dump for ANOIS1 (1 MHz atmospheric noise).  A real REDMAP
C     loads the /TWO/ noise maps (read by NOISY as the P(29,16,8) overlay) and
C     F2D; the driver injects the receiver mean time GMTR and location
C     RLAT/RLONG/RLONGD, calls ANOIS1, and dumps the two block noise values
C     ATNU/ATNY plus the block indices/times KJ/JK/CC/TM.  IZF forces the
C     no-data-base fallback (ATNU=ATNY=0).
      use voacapl_defs
      use crun_directory
      COMMON /ccoeff/ COEFF
      CHARACTER COEFF*4
      COMMON /FONE/ ANEW(3), BNEW(3), ACHI(2), BCHI(2)
      COMMON /ONE/ IA(6), IB(6), IKIM(10,6), ESLCOF(5,55),
     1             ESMCOF(7,61), ESUCOF(5,55), F2COF(13,76),
     2             FM3COF(9,49), ERCOF(9,22)
      COMMON /TWO/ F2D(16,6,6), FAKP(29,16,6), FAKMAP(29,16),
     1             HMYM(29,16), FAKABP(2,6), ABMAP(2,3),
     2             DUD(5,12,5), FAM(14,12), SYS(9,16,6), PERR(9,4,6)
      COMMON /CON /D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON / TIME / IT, GMT, UTIME(24), GMTR, XLMT(24), ITIM, JTX
      COMMON/ANOIS/ATNU,ATNY,CC,TM,RCNSE,DU,DL,SIGM,SYGU,SYGL,KJ,JK
      DIMENSION IIA(6), IIB(6)
      CHARACTER HOMED*200
      INTEGER MONTHIN, IZF
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
      DATA D2R/.01745329251/, R2D/57.295779513/
      COEFF = 'CCIR'
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
      OPEN(10,FILE='anois1_cases.csv',STATUS='OLD')
      OPEN(11,FILE='anois1_dump.csv')
      WRITE(11,'(A)') 'atnu,atny,cc,tm,kj,jk'
10    READ(10,*,END=99) MONTHIN, SSNIN, GMTRI, RLATI, RLONGI, RLONGDI, IZF
      MONTH=MONTHIN
      SSN=SSNIN
      DO 15 I=1,6
      IA(I)=IIA(I)
15    IB(I)=IIB(I)
      CALL REDMAP(SSN, MONTH)
      IF(IZF.NE.0) F2D(1,1,1) = -1.0
      GMTR=GMTRI
      RLAT=RLATI
      RLONG=RLONGI
      RLONGD=RLONGDI
      CALL ANOIS1
      WRITE(11,20) ATNU,ATNY,CC,TM,FLOAT(KJ),FLOAT(JK)
20    FORMAT(ES18.9E3,5(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
