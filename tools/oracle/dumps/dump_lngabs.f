      PROGRAM DUMPLNGABS
C     Tier-B oracle dump for BABS + TABS + CONVH, the three pure leaves
C     LNGPAT calls to price its geometry.
C
C     All three read only /FRQ/ FREQ, /CON/, /GEOG/ GYZ(3) and /SIGD/ ACAV
C     and write only their output arguments, so driving them together costs
C     nothing: no routine can disturb another's inputs.  DEL and HP are
C     shared across all three exactly as LNGPAT shares them.
C
C     /CON/ is DATA-initialised here rather than linked from blkdat.for --
C     BABS and TABS need only RZ and PIO2, and pulling in blkdat would drag
C     its whole DATA set (and forbid this program initialising any of the
C     same commons) for two constants.  The values are blkdat's verbatim.
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON /CON /D2R, DCL, GAMA, PI, PI2, PIO2, R2D, RZ, VOFL
      COMMON / GEOG / GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5),
     1 ARTIC(5), SIGPAT(5), EPSPAT(5)
      COMMON/SIGD/ DSL,ASM,DSU,AGLAT,DSLF,ASMF,DSUF,ACAV,FEAV,AFE,BFE
     A ,HNU,HTLOSS,XNUZ,XVE
      DATA D2R/0.01745329251/, DCL/1.28/, GAMA/0.57721566/,
     A     PI/3.1415926/, PI2/6.283185307/, PIO2/1.570796326/,
     B     R2D/57.295779513/, RZ/6370./, VOFL/299.79246/
      OPEN(10,FILE='lngabs_cases.csv',STATUS='OLD')
      OPEN(11,FILE='lngabs_dump.csv')
      WRITE(11,'(A)') 'af,am,ch,ptot'
20    READ(10,*,END=99) FREQI,GYZ3I,ACAVI,DELI,HPI,DFI,DMI,GDI,PHEI,RAYI
      FREQ   = FREQI
      GYZ(3) = GYZ3I
      ACAV   = ACAVI
C          ---- distinctive sentinels: "untouched" must not read as 0 ----
      AF   = 12345.
      AM   = -8888.
      CH   = -7777.
      PTOT = -6666.
      CALL BABS(DELI,HPI,DFI,AF)
      CALL TABS(DELI,HPI,DMI,AM)
      CALL CONVH(GDI,PHEI,DELI,HPI,RAYI,CH,PTOT)
      WRITE(11,30) AF,AM,CH,PTOT
30    FORMAT(ES18.9E3,3(',',ES18.9E3))
      GO TO 20
99    CLOSE(10)
      CLOSE(11)
      END
