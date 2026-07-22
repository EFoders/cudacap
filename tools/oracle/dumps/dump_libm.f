      PROGRAM DUMPLIBM
C     Raw REAL*4 intrinsic dump: forces gfortran to emit runtime libm
C     calls (arguments come from READ, so nothing constant-folds) and
C     records the bits.  This is the emulation target for f32math — the
C     glibc flt-32 sinf/cosf/acosf/atanf actually linked by the oracle.
      REAL X, SC, CC, AC, TC
      INTEGER IX, IS, ICC, IA, IT
      EQUIVALENCE (X,IX),(SC,IS),(CC,ICC),(AC,IA),(TC,IT)
      OPEN(10,FILE='libm_cases.csv',STATUS='OLD')
      OPEN(11,FILE='libm_dump.csv')
      WRITE(11,'(A)') 'xbits,sinbits,cosbits,acosbits,atanbits'
10    READ(10,*,END=99) X
      SC = SIN(X)
      CC = COS(X)
      IF (X .GE. -1.0 .AND. X .LE. 1.0) THEN
        AC = ACOS(X)
      ELSE
        AC = 0.0
      END IF
      TC = ATAN(X)
      WRITE(11,20) IX, IS, ICC, IA, IT
20    FORMAT(I0,4(',',I0))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
