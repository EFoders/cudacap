      PROGRAM DUMPGENFAM
C     Tier-B oracle dump for GENFAM (atmospheric-noise frequency dependence).
C
C     A real REDMAP loads the FAM(14,12) and DUD(5,12,5) tables into /TWO/ --
C     both come straight out of the month's coefficient file with no SSN
C     blending, so SSN is fixed and only MONTH selects which tables are in
C     play.  GENFAM then takes explicit arguments, so nothing else needs
C     setting up: this is the cleanest routine in the noise chain to pin.
C
C     Y2 is used for its SIGN ONLY (negative shifts the local-time block by 6
C     into the southern-hemisphere half of the tables), so the case file
C     sweeps both signs rather than realistic latitudes.
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
      DIMENSION IIA(6), IIB(6)
      CHARACTER HOMED*200
      DATA IIA / 1,276,703,978,1966,2407 /
      DATA IIB / 5, 7, 5, 13, 9, 9 /
      SSN = 100.0
      COEFF = 'CCIR'
      CALL GET_ENVIRONMENT_VARIABLE('HOME', HOMED)
      root_directory = trim(HOMED)//'/itshfbc'
      OPEN(10,FILE='genfam_cases.csv',STATUS='OLD')
      OPEN(11,FILE='genfam_dump.csv')
      WRITE(11,'(A)') 'fa,dua,dla,dms,dus,dls'
      MPREV = -1
10    READ(10,*,END=99) MONTHIN, Y2IN, IBLKIN, FREQIN, ZIN
      IF(MONTHIN.NE.MPREV) THEN
         MONTH=MONTHIN
         DO 15 I=1,6
         IA(I)=IIA(I)
15       IB(I)=IIB(I)
         CALL REDMAP(SSN, MONTH)
         MPREV = MONTHIN
      END IF
      CALL GENFAM(Y2IN,IBLKIN,FREQIN,ZIN,FA,DUA,DLA,DMS,DUS,DLS)
      WRITE(11,20) FA,DUA,DLA,DMS,DUS,DLS
20    FORMAT(ES18.9E3,5(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
