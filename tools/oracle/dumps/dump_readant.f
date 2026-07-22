      PROGRAM DUMPRDANT
C     Tier-B oracle dump for READANT's table paths (jant 11 / 13 / 14).
C
C     The driver replicates readant.for's READ statements verbatim rather than
C     linking the routine itself: READANT's remaining body is filename/path
C     plumbing (rblankc, run_directory, PATH_SEPARATOR, the voacapl_defs and
C     crun_directory modules) that is deliberately not ported, and its type
C     13/14 payload goes out through direct-access scratch files.  What is
C     under test is the record parsing -- format reversion across the 91-value
C     tables, T-editing, BLANK='NULL' and short-record blank padding -- so
C     letting gfortran execute the original FORMATs on the original sample
C     files is the oracle that matters.
C
C     readant_cases.txt holds "tag path" lines; output is one flat CSV.
      CHARACTER*200 LINE, FPATH
      CHARACTER*8 TAG
      CHARACTER*70 DESCR
      DIMENSION PARM(20), GAINS(91)
      OPEN(9,FILE='readant_cases.txt',STATUS='OLD')
      OPEN(11,FILE='readant_dump.csv')
      WRITE(11,'(A)') 'tag,kind,i,j,value'
5     READ(9,'(A)',END=99) LINE
      IF(LINE(1:1).EQ.'#' .OR. LINE(1:1).EQ.' ') GO TO 5
      NSP = INDEX(LINE,' ')
      TAG = LINE(1:NSP-1)
      FPATH = LINE(NSP+1:)
      OPEN(10,FILE=TRIM(FPATH),STATUS='OLD')
      REWIND(10)
C          header: description, parameter count, then N free-form parameters
      READ(10,'(A)') DESCR
      READ(10,*) N
      DO 10 I=1,N
      READ(10,*) PARM(I)
10    CONTINUE
      DO 11 I=N+1,20
11    PARM(I)=0.
      JANT=NINT(PARM(2))
      DO 12 I=1,20
12    WRITE(11,50) TRIM(TAG),'parm',I,0,PARM(I)
      IF(JANT.EQ.11) THEN
C          type 11 = gain table at 91 elevation angles
         READ(10,*) GAINS
         DO 20 J=1,91
20       WRITE(11,50) TRIM(TAG),'t11',1,J,GAINS(J)
      ELSE IF(JANT.EQ.13) THEN
C          type 13 = gain table at 360x91 values
         DO 32 IANG=0,359
         READ(10,'(9X,10F7.3)') GAINS
         DO 31 J=1,91
31       WRITE(11,50) TRIM(TAG),'t13',IANG+1,J,GAINS(J)
32       CONTINUE
      ELSE IF(JANT.EQ.14) THEN
C          type 14 = gain table at 30x(efficiency + 91 values)
         DO 42 IFREQ=1,30
         READ(10,'(2X,F6.1,(T10,10F7.3))') EFF,GAINS
         WRITE(11,50) TRIM(TAG),'t14e',IFREQ,0,EFF
         DO 41 J=1,91
41       WRITE(11,50) TRIM(TAG),'t14g',IFREQ,J,GAINS(J)
42       CONTINUE
      ELSE
         WRITE(*,'('' skipping '',A,'' -- type '',I0,'' is not tabulated'')')
     +         TRIM(TAG), JANT
      END IF
50    FORMAT(A,',',A,',',I0,',',I0,',',ES18.9E3)
      CLOSE(10)
      GO TO 5
99    CLOSE(9)
      CLOSE(11)
      END
