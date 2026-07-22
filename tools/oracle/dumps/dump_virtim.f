      PROGRAM DUMPVIRTIM
C     Tier-B oracle dump driver for REDMAP+VIRTIM (Module 2).
C     Reads virtim_cases.csv (icoeff,month,ssn,gmt; icoeff 0=CCIR
C     1=URSI), loads the coefficient files from $HOME/itshfbc via the
C     real REDMAP (SSN blend), runs VIRTIM's diurnal Fourier collapse,
C     writes all 318 AB coefficients.
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
      COMMON /TWO_AB/ AB(318)
      COMMON /TIME/ IT, GMT, UTIME(24), GMTR, XLMT(24), ITIM, JTX
      COMMON /CON/ D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      DIMENSION IIA(6), IIB(6)
      CHARACTER HOMED*200
      INTEGER ICOEFF, MONTH, I
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
      DATA RZ/6370./, PI/3.1415926/, VOFL/299.79246/, DCL/1.28/
      DATA GAMA/0.57721566/, PI2/6.283185307/, PIO2/1.570796326/
      DATA D2R/.01745329251/, R2D/57.295779513/
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
      OPEN(10,FILE='virtim_cases.csv',STATUS='OLD')
      OPEN(11,FILE='virtim_dump.csv')
      WRITE(11,'(A)') 'ab_1_to_318'
10    READ(10,*,END=99) ICOEFF, MONTH, SSN, GMTIN
      COEFF='CCIR'
      IF(ICOEFF.EQ.1) COEFF='URSI'
      DO 15 I=1,6
      IA(I)=IIA(I)
15    IB(I)=IIB(I)
      CALL REDMAP(SSN, MONTH)
      GMT=GMTIN
      DO 16 I=1,318
16    AB(I)=0.0
      CALL VIRTIM
      WRITE(11,20) (AB(I),I=1,318)
20    FORMAT(ES18.9E3,317(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
