      PROGRAM DUMPSANG
C     Tier-B oracle dump driver for SANG. Reads sang_cases.csv
C     (gcdkm,amind), calls SANG, writes NANG. The ANG table DATA is
C     copied verbatim from blkdat.for.
      COMMON/RAYS/ANG(40),IFOB(40,30,5),NANG
      COMMON /DON/ ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR,
     1 GCD, GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR,
     2 RLAT, RLATD, RLONG, RLONGD, BRTD, FLUX, ULAT, ULATD, ULONG,
     3 ULONGD, SSN, D90R, D50R, D10R, D90S, D50S, D10S
      DATA ANG/0., 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0
     1,14.,16.,18.,20.,22.,24.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.
     2 ,46.,48.,50.,52.,54.,56.,60.,65.,70.,75.,80.,85.,89.99/
      OPEN(10,FILE='sang_cases.csv',STATUS='OLD')
      OPEN(11,FILE='sang_dump.csv')
      WRITE(11,'(A)') 'nang'
10    READ(10,*,END=99) GCDKM, AMIND
      NANG=0
      CALL SANG
      WRITE(11,'(I2)') NANG
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
