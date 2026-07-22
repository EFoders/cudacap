      PROGRAM DUMPPRBMUF
C     Tier-B oracle dump for the PRBMUF probability function.  PRBMUF reads
C     only SIGL(IL)/SIGU(IL) from /MUFS/, so the driver injects those per case
C     and calls PRBMUF(FMHZ,FGO,FSET,IL) directly.  Cases span the FSET<=0 step
C     branch, the FSET>0 branch on both sides of the median (SIGL vs SIGU), the
C     |Z| clamp at 5, and the SIG floor at 0.001.
      COMMON/MUFS/EMUF(24),F1MUF(24),F2MUF(24),ESMUF(24),ALLMUF(24),FOT
     A (24),XLUF(24),HPF(24),ANGMUF(24),MODMUF,SIGL(4),SIGU(4),DELMUF(4)
     B ,HPMUF(4),HTMUF(4),FVMUF(4),AFMUF(4),NHOPMF(4),YFOT(4),YHPF(4)
     C ,YMUF(4)
      REAL PRBMUF
      INTEGER ILIN
      OPEN(10,FILE='prbmuf_cases.csv',STATUS='OLD')
      OPEN(11,FILE='prbmuf_dump.csv')
      WRITE(11,'(A)') 'prbmuf'
10    READ(10,*,END=99) FMHZ,FGO,FSET,ILIN,SIGLIL,SIGUIL
      IL=ILIN
      SIGL(IL)=SIGLIL
      SIGU(IL)=SIGUIL
      P=PRBMUF(FMHZ,FGO,FSET,IL)
      WRITE(11,20) P
20    FORMAT(ES18.9E3)
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
