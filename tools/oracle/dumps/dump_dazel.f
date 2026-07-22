      PROGRAM DUMPDAZEL
C     Tier-B oracle dump driver for DAZEL (see PLAN.md section 3B).
C     Reads dazel_cases.csv (mode,tlat,tlon,tht,rlat,rlon,rht,taz,dgc),
C     calls DAZEL, writes every COMMON/AZEL/ field at 9 significant
C     digits (full REAL*4 round-trip precision).
      COMMON/AZEL/ ZTLAT,ZTLON,ZTHT,ZRLAT,ZRLON,ZRHT,ZTAZ,ZRAZ,
     * ZTELV,ZRELV,ZD,ZDGC,ZTAKOF,ZRAKOF
      INTEGER MODE
      OPEN(10,FILE='dazel_cases.csv',STATUS='OLD')
      OPEN(11,FILE='dazel_dump.csv')
      WRITE(11,'(A,A)') 'mode,ztlat,ztlon,ztht,zrlat,zrlon,zrht,',
     *'ztaz,zraz,ztelv,zrelv,zd,zdgc,ztakof,zrakof'
10    READ(10,*,END=99) MODE,ZTLAT,ZTLON,ZTHT,ZRLAT,ZRLON,ZRHT,
     *                  ZTAZ,ZDGC
      ZRAZ=0.0
      ZTELV=0.0
      ZRELV=0.0
      ZD=0.0
      ZTAKOF=0.0
      ZRAKOF=0.0
      CALL DAZEL(MODE)
      WRITE(11,20) MODE,ZTLAT,ZTLON,ZTHT,ZRLAT,ZRLON,ZRHT,ZTAZ,ZRAZ,
     *             ZTELV,ZRELV,ZD,ZDGC,ZTAKOF,ZRAKOF
20    FORMAT(I1,14(',',ES18.9E3))
      GO TO 10
99    CLOSE(10)
      CLOSE(11)
      END
