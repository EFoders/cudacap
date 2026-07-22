      SUBROUTINE LUFFY(IPFLAG)
C     Tier-A replay capture for LUFFY, shimmed INTO the real program.
C
C     Every other dump in this directory injects state into one routine.
C     LUFFY cannot be driven that way: it builds its own ionosphere from the
C     coefficient maps, so injecting its inputs would mean injecting most of
C     the program.  Instead this file REPLACES LUFFY in the link, records the
C     state at entry, calls the real routine under the name LUFFYR, and
C     records the state at exit.  The rest of voacapl -- card reading, the
C     map interpolation, CURMUF, the hour loop -- runs untouched, so the
C     captured state is exactly what a real deck produces.
C
C     The runner builds LUFFYR by copying reference/.../luffy.for and
C     renaming its SUBROUTINE statement.  reference/ itself is never touched.
C
C     The antenna patterns ARE captured, once per deck rather than per hour,
C     because ANTCALC derives OFFAZIM from the beam heading and the path
C     bearing and nothing short of running it reproduces that.  They go to
C     their own file, one row per (antenna, frequency), so the per-hour CSV
C     stays readable.
C
C     NOT CAPTURED: the coefficient maps, which REDMAP loads Python-side from
C     the same files and is already validated against its own dump; and /SON/
C     MODE and MDL, which are CHARACTER and feed only the output tables.
      COMMON / ALPHA / IMON(12), ITRAN(2), IRCVR(2), LBMAP(2), MODE(13),
     A MODER(13), ITLAT, ITLONG, IRLAT, IRLONG, IRLATU, IRLONGU, NYEAR
      CHARACTER IMON*3, NYEAR*5, ITRAN*10, IRCVR*10, LBMAP*10, ITLAT*1,
     A ITLONG*1, IRLAT*1, IRLONG*1, IRLATU*1, IRLONGU*1, MODE*2, MODER*2
      COMMON / CONTRL / IELECT(3), KTOUT(12), MONTHS(12), SUNSP(12),
     A IANTOU, ICARD, INTEG, IRED, ISOUT, ISPROC, ISSN, ITYPE, JDASH,
     B JFREQ, JLONG, KCARD, KRUN, MAPIN, MAXNAM, MONOLD, MOREM, MORES,
     C NUMNAM, NUPROC, MAXMET, MSPEC, M100
      COMMON /DON /ALATD, AMIN, AMIND, BTR, BTRD, DLONG, DMP, ERTR, GCD,
     1 GCDKM, PMP, PWR, TLAT, TLATD, TLONG, TLONGD, RSN, SIGTR, RLAT,
     2 RLATD,RLONG,RLONGD,BRTD,FLUX,ULAT,ULATD,ULONG,ULONGD,SSN,D90R,
     3 D50R,D10R,D90S,D50S,D10S
      COMMON / ES / FS(3,5), HS(5)
      COMMON/FRQ/FREA(13),FREL(29),FREQ,JMODE,ITXRCP(2)
      COMMON / GEOG / GYZ(5), RAT(5), GMDIP(5), CLCK(5), ABIY(5),
     1 ARTIC(5), SIGPAT(5), EPSPAT(5)
      COMMON / ION / IHRE, IHRO, IHRS, LUFP, METHOD, NOISE, NPSL
      COMMON / MFAC / F2M3(5),HPF2(5),ZENANG(5),ZENMAX(5),IEDP,FSECV(3)
      COMMON/MUFS/EMUF(24),F1MUF(24),F2MUF(24),ESMUF(24),ALLMUF(24),FOT
     A (24),XLUF(24),HPF(24),ANGMUF(24),MODMUF,SIGL(4),SIGU(4),DELMUF(4)
     B ,HPMUF(4),HTMUF(4),FVMUF(4),AFMUF(4),NHOPMF(4),YFOT(4),YHPF(4)
     C ,YMUF(4)
      COMMON / OUTPRT / LINBOT(26), LINBD(14), LINTOP(15), LINTP(14),
     A GRPTYP, JOUT, LINBYP, LINES, LINMAX, LINTYP, LPAGES, NLINE
      COMMON / PSCA / PSC(4), PSCB(4), IPSC
      COMMON/RAYS/ANG(40),IFOB(40,30,5),NANG
      COMMON /RON /CLAT(5), CLONG(5), GLAT(5), RD(5), FI(3,5), YI(3,5),
     1HI(3,5), HPRIM(30,5), HTRUE(30,5), FVERT(30,5),KM,KFX, AFAC(30,5),
     2HTR(50,3), FNSQ(50,3)
      COMMON / SON / ANGLE(13), ANGLER(13), CPROB(13), DBLOS(13),
     A DBLOSL(13), DBLOSU(13), DBU(13), DELAY(13), DBW(13), NHP(13),
     B XNYNOIS(13), PROBMP(13), RELIAB(13), SNDB(13), SNPR(13),
     C SNRLW(13), SNRUP(13), SPROB(13), VHIGH(13), RNEFF(13),MDL(13)
      CHARACTER MDL*1
      COMMON /SSP /SUN (2, 12), MONTH
      COMMON / TIME / IT, GMT, UTIME(24), GMTR, XLMT(24), ITIM, JTX
      common /pantenna/ pwrkw(20),pwrdba(20)
      DIMENSION V(400), W(300)
      LOGICAL FIRST
      SAVE FIRST
      DATA FIRST/.TRUE./

      IF(FIRST)THEN
        OPEN(81,FILE='luffy_in.csv')
        OPEN(82,FILE='luffy_out.csv')
        CALL LUFHDR
        CALL LUFANT
        FIRST=.FALSE.
      ENDIF

C          ================= ENTRY STATE =================
      N = 0
C          -- the key: which deck, month, hour and model --
      CALL PUT(V,N,FLOAT(MONTH))
      CALL PUT(V,N,SSN)
      CALL PUT(V,N,FLOAT(IPFLAG))
      CALL PUT(V,N,FLOAT(METHOD))
      CALL PUT(V,N,FLOAT(MSPEC))
      CALL PUT(V,N,FLOAT(LUFP))
      CALL PUT(V,N,FLOAT(NOISE))
      CALL PUT(V,N,FLOAT(NPSL))
      CALL PUT(V,N,FLOAT(LINBOT(14)))
      CALL PUT(V,N,FLOAT(KM))
      CALL PUT(V,N,FLOAT(KFX))
      CALL PUT(V,N,FLOAT(IEDP))
      CALL PUT(V,N,FLOAT(NANG))
      CALL PUT(V,N,FLOAT(IT))
      CALL PUT(V,N,GMT)
      CALL PUT(V,N,GMTR)
C          -- /DON/ path geometry and circuit requirements --
      CALL PUT(V,N,GCD)
      CALL PUT(V,N,GCDKM)
      CALL PUT(V,N,AMIND)
      CALL PUT(V,N,RSN)
      CALL PUT(V,N,DMP)
      CALL PUT(V,N,PMP)
      CALL PUT(V,N,PWR)
      CALL PUT(V,N,RLAT)
      CALL PUT(V,N,RLONG)
      CALL PUT(V,N,RLONGD)
      CALL PUT(V,N,FLUX)
      CALL PUT(V,N,pwrkw(1))
      CALL PUT(V,N,pwrdba(1))
C          -- /MUFS/ at this hour, then the per-layer arrays --
      CALL PUT(V,N,EMUF(IT))
      CALL PUT(V,N,F1MUF(IT))
      CALL PUT(V,N,F2MUF(IT))
      CALL PUT(V,N,ESMUF(IT))
      CALL PUT(V,N,ALLMUF(IT))
      CALL PUT(V,N,FOT(IT))
      CALL PUT(V,N,HPF(IT))
      CALL PUT(V,N,ANGMUF(IT))
      CALL PUT(V,N,FLOAT(MODMUF))
      DO 10 I=1,4
      CALL PUT(V,N,SIGL(I))
      CALL PUT(V,N,SIGU(I))
      CALL PUT(V,N,DELMUF(I))
      CALL PUT(V,N,HPMUF(I))
      CALL PUT(V,N,HTMUF(I))
      CALL PUT(V,N,FVMUF(I))
      CALL PUT(V,N,AFMUF(I))
      CALL PUT(V,N,FLOAT(NHOPMF(I)))
      CALL PUT(V,N,YFOT(I))
      CALL PUT(V,N,YHPF(I))
10    CALL PUT(V,N,YMUF(I))
C          -- /RON/ sample areas and layer parameters --
      DO 12 K=1,5
      CALL PUT(V,N,CLAT(K))
      CALL PUT(V,N,CLONG(K))
      CALL PUT(V,N,GLAT(K))
12    CALL PUT(V,N,RD(K))
      DO 14 K=1,5
      DO 14 L=1,3
      CALL PUT(V,N,FI(L,K))
      CALL PUT(V,N,YI(L,K))
14    CALL PUT(V,N,HI(L,K))
C          -- /ES/, /MFAC/, /GEOG/, /PSCA/ --
      DO 16 K=1,5
      DO 15 L=1,3
15    CALL PUT(V,N,FS(L,K))
16    CALL PUT(V,N,HS(K))
      DO 18 K=1,5
      CALL PUT(V,N,F2M3(K))
      CALL PUT(V,N,HPF2(K))
      CALL PUT(V,N,ZENANG(K))
18    CALL PUT(V,N,ZENMAX(K))
      DO 19 L=1,3
19    CALL PUT(V,N,FSECV(L))
      DO 20 K=1,5
      CALL PUT(V,N,GYZ(K))
      CALL PUT(V,N,RAT(K))
      CALL PUT(V,N,GMDIP(K))
      CALL PUT(V,N,CLCK(K))
      CALL PUT(V,N,ABIY(K))
      CALL PUT(V,N,ARTIC(K))
      CALL PUT(V,N,SIGPAT(K))
20    CALL PUT(V,N,EPSPAT(K))
      DO 22 L=1,4
22    CALL PUT(V,N,PSC(L))
      DO 23 L=1,3
23    CALL PUT(V,N,FLOAT(IELECT(L)))
C          -- /FRQ/ the frequency list, and /RAYS/ the angle scan --
      DO 24 I=1,14
24    CALL PUT(V,N,FREL(I))
      DO 26 I=1,40
26    CALL PUT(V,N,ANG(I))
      WRITE(81,90) (V(I),I=1,N)

C          ================= THE REAL ROUTINE =================
      CALL LUFFYR(IPFLAG)

C          ================= EXIT STATE =================
      M = 0
      DO 40 I=1,13
      CALL PUT(W,M,ANGLE(I))
      CALL PUT(W,M,ANGLER(I))
      CALL PUT(W,M,CPROB(I))
      CALL PUT(W,M,DBLOS(I))
      CALL PUT(W,M,DBLOSL(I))
      CALL PUT(W,M,DBLOSU(I))
      CALL PUT(W,M,DBU(I))
      CALL PUT(W,M,DELAY(I))
      CALL PUT(W,M,DBW(I))
      CALL PUT(W,M,FLOAT(NHP(I)))
      CALL PUT(W,M,XNYNOIS(I))
      CALL PUT(W,M,PROBMP(I))
      CALL PUT(W,M,RELIAB(I))
      CALL PUT(W,M,SNDB(I))
      CALL PUT(W,M,SNPR(I))
      CALL PUT(W,M,SNRLW(I))
      CALL PUT(W,M,SNRUP(I))
      CALL PUT(W,M,SPROB(I))
      CALL PUT(W,M,VHIGH(I))
40    CALL PUT(W,M,RNEFF(I))
      CALL PUT(W,M,XLUF(IT))
      CALL PUT(W,M,FLOAT(JMODE))
      WRITE(82,90) (W(I),I=1,M)
90    FORMAT(ES18.9E3,999(',',ES18.9E3))
      RETURN
      END
C--------------------------------
      SUBROUTINE LUFANT
C     /cantenna/ and /pantenna/ as ANTCALC left them: constant across the
C     hour loop, so written once.  One row per (antenna, frequency), with the
C     91 elevation gains as columns.  OFFAZIM is the reason this is captured
C     at all -- it is the angle between the antenna's main beam and the path,
C     which only ANTCALC's card processing knows.
      common /cantenna/ numants,iats(20),anttype(20),antname(20),
     +                  xfqs(20),xfqe(20),designfreq(20),antfile(20),
     +                  beammain(20),offazim(20),cond(20),diel(20),
     +                  array(30,91,20),aeff(30,20)
      character anttype*10,antname*70,antfile*24
      common /pantenna/ pwrkw(20),pwrdba(20)
      CHARACTER H*1200
      OPEN(83,FILE='luffy_ant.csv')
      H=' '
      CALL ADD(H,'iant,itr,ifreq,xfqs,xfqe,beammain,offazim,cond,diel')
      CALL ADD(H,'pwrkw,pwrdba,aeff')
      DO 5 J=1,91
5     CALL ADDI(H,'g',J)
      WRITE(83,'(A)') H(2:LEN_TRIM(H))
      NA = NUMANTS
      IF(NA.gt.20) NA = 20
      DO 20 IA=1,NA
      DO 20 IF=1,30
      WRITE(83,30) FLOAT(IA),FLOAT(IATS(IA)),FLOAT(IF),XFQS(IA),
     A  XFQE(IA),BEAMMAIN(IA),OFFAZIM(IA),COND(IA),DIEL(IA),
     B  PWRKW(IA),PWRDBA(IA),AEFF(IF,IA),(ARRAY(IF,J,IA),J=1,91)
20    CONTINUE
30    FORMAT(ES18.9E3,102(',',ES18.9E3))
      CLOSE(83)
      RETURN
      END
C--------------------------------
      SUBROUTINE PUT(A,N,X)
C     Append X to A and bump the cursor.  Keeps the caller's value list a
C     flat sequence, so the Python column order is the order written here.
      DIMENSION A(*)
      N = N + 1
      A(N) = X
      RETURN
      END
C--------------------------------
      SUBROUTINE LUFHDR
C     CSV headers.  The names are the Fortran ones lowercased; array slots
C     carry their subscripts in Fortran order, slowest-varying first.
      CHARACTER H*8000
      H=' '
      CALL ADD(H,'month,ssn,ipflag,method,mspec,lufp,noise,npsl,linbot14')
      CALL ADD(H,'km,kfx,iedp,nang,it,gmt,gmtr')
      CALL ADD(H,'gcd,gcdkm,amind,rsn,dmp,pmp,pwr,rlat,rlong,rlongd')
      CALL ADD(H,'flux,pwrkw,pwrdba')
      CALL ADD(H,'emuf,f1muf,f2muf,esmuf,allmuf,fot,hpf,angmuf,modmuf')
      DO 10 I=1,4
      CALL ADDI(H,'sigl',I)
      CALL ADDI(H,'sigu',I)
      CALL ADDI(H,'delmuf',I)
      CALL ADDI(H,'hpmuf',I)
      CALL ADDI(H,'htmuf',I)
      CALL ADDI(H,'fvmuf',I)
      CALL ADDI(H,'afmuf',I)
      CALL ADDI(H,'nhopmf',I)
      CALL ADDI(H,'yfot',I)
      CALL ADDI(H,'yhpf',I)
10    CALL ADDI(H,'ymuf',I)
      DO 12 K=1,5
      CALL ADDI(H,'clat',K)
      CALL ADDI(H,'clong',K)
      CALL ADDI(H,'glat',K)
12    CALL ADDI(H,'rd',K)
      DO 14 K=1,5
      DO 14 L=1,3
      CALL ADDJ(H,'fi',L,K)
      CALL ADDJ(H,'yi',L,K)
14    CALL ADDJ(H,'hi',L,K)
      DO 16 K=1,5
      DO 15 L=1,3
15    CALL ADDJ(H,'fs',L,K)
16    CALL ADDI(H,'hs',K)
      DO 18 K=1,5
      CALL ADDI(H,'f2m3',K)
      CALL ADDI(H,'hpf2',K)
      CALL ADDI(H,'zenang',K)
18    CALL ADDI(H,'zenmax',K)
      DO 19 L=1,3
19    CALL ADDI(H,'fsecv',L)
      DO 20 K=1,5
      CALL ADDI(H,'gyz',K)
      CALL ADDI(H,'rat',K)
      CALL ADDI(H,'gmdip',K)
      CALL ADDI(H,'clck',K)
      CALL ADDI(H,'abiy',K)
      CALL ADDI(H,'artic',K)
      CALL ADDI(H,'sigpat',K)
20    CALL ADDI(H,'epspat',K)
      DO 22 L=1,4
22    CALL ADDI(H,'psc',L)
      DO 23 L=1,3
23    CALL ADDI(H,'ielect',L)
      DO 24 I=1,14
24    CALL ADDI(H,'frel',I)
      DO 26 I=1,40
26    CALL ADDI(H,'ang',I)
      WRITE(81,'(A)') H(2:LEN_TRIM(H))

      H=' '
      DO 40 I=1,13
      CALL ADDI(H,'angle',I)
      CALL ADDI(H,'angler',I)
      CALL ADDI(H,'cprob',I)
      CALL ADDI(H,'dblos',I)
      CALL ADDI(H,'dblosl',I)
      CALL ADDI(H,'dblosu',I)
      CALL ADDI(H,'dbu',I)
      CALL ADDI(H,'delay',I)
      CALL ADDI(H,'dbw',I)
      CALL ADDI(H,'nhp',I)
      CALL ADDI(H,'xnynois',I)
      CALL ADDI(H,'probmp',I)
      CALL ADDI(H,'reliab',I)
      CALL ADDI(H,'sndb',I)
      CALL ADDI(H,'snpr',I)
      CALL ADDI(H,'snrlw',I)
      CALL ADDI(H,'snrup',I)
      CALL ADDI(H,'sprob',I)
      CALL ADDI(H,'vhigh',I)
40    CALL ADDI(H,'rneff',I)
      CALL ADD(H,'xluf,jmode')
      WRITE(82,'(A)') H(2:LEN_TRIM(H))
      RETURN
      END
C--------------------------------
      SUBROUTINE ADD(H,S)
      CHARACTER H*(*), S*(*)
      WRITE(H(LEN_TRIM(H)+1:),'(A,A)') ',', S
      RETURN
      END
C--------------------------------
      SUBROUTINE ADDI(H,S,I)
      CHARACTER H*(*), S*(*)
      WRITE(H(LEN_TRIM(H)+1:),'(A,A,I2.2)') ',', S, I
      RETURN
      END
C--------------------------------
      SUBROUTINE ADDJ(H,S,I,J)
      CHARACTER H*(*), S*(*)
      WRITE(H(LEN_TRIM(H)+1:),'(A,A,I1,I1)') ',', S, I, J
      RETURN
      END
C--------------------------------
