!> \file scalability_benchmark_fortran.F90
!!
!! Scalability benchmark for the Fortran Kessler microphysics scheme.
!! JD-driver variant: uses physics-based inputs mirroring JD_kessler_driver.F90
!! instead of uniform synthetic fills.
!!
!! Sweeps ncol at fixed nz=56 and times kessler_run with SYSTEM_CLOCK.
!! Per-column scaling factor arr(i) ~ N(mean=1, std=0.1) via Box-Muller,
!! matching the Fortran driver's random-number approach with a fixed seed.
!!
!! Timing methodology
!! ------------------
!!   - One warmup call per ncol (includes any first-call overhead)
!!   - N_REPS=5 timed calls, inout arrays re-initialized before each
!!   - Median of N_REPS reported as the benchmark time
!!
!! Outputs
!! -------
!!   _compare_results/outputs/scalability/fortran_scalability_results.json
!!   (path relative to the kessler project root; the binary must be run there)
!!
!! Compilation (from the kessler project root)
!! -------------------------------------------
!!   gfortran -O2 -ffree-line-length-none \
!!       -o data/exp2_jd/src/scalability_benchmark_fortran \
!!       data/exp2_jd/src/ccpp_kinds.F90 \
!!       data/exp2_jd/src/kessler.F90 \
!!       data/exp2_jd/src/scalability_benchmark_fortran.F90
!!
!! Run (from the kessler project root)
!! -----------------------------------
!!   mkdir -p _compare_results/outputs/scalability
!!   ./data/exp2_jd/src/scalability_benchmark_fortran
!!
!! Or submit ../LAFT/pbsJobs/compare_scalability_fortran.sh, which does both.
!!

program scalability_benchmark_fortran

   use ccpp_kinds, only: kind_phys
   use kessler,    only: kessler_init, kessler_run

   implicit none

   ! ---------------------------------------------------------------------------
   ! Configuration — mirrors JD_kessler_driver.F90
   ! ---------------------------------------------------------------------------
   integer,  parameter :: N_SWEEP  = 10
   integer,  parameter :: NZ       = 56
   integer,  parameter :: N_REPS   = 5
   integer,  parameter :: LYR_SURF = 1
   integer,  parameter :: LYR_TOA  = NZ
   real(kind_phys), parameter :: DT = 60.0_kind_phys

   integer, dimension(N_SWEEP), parameter :: NCOL_SWEEP = &
        [50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000, 1000000]

   ! kessler_init constants — match JD driver
   real(kind_phys), parameter :: LV_IN    = 2.5e6_kind_phys
   real(kind_phys), parameter :: PREF_IN  = 100000.0_kind_phys   ! Pa (1000 hPa)
   real(kind_phys), parameter :: RHOQR_IN = 1000.0_kind_phys

   ! ---------------------------------------------------------------------------
   ! Timing variables
   ! ---------------------------------------------------------------------------
   integer(kind=8) :: t_start, t_end, cr
   real(kind=8)    :: times(N_REPS), warmup_s
   real(kind=8)    :: median_s, mean_s, std_s

   ! ---------------------------------------------------------------------------
   ! Kessler arrays (allocatable — resized each ncol)
   ! Layout: (ncol, nz) — Fortran column-major, kessler_run accesses (col, klev)
   ! ---------------------------------------------------------------------------
   real(kind_phys), allocatable :: cpair(:,:), rair(:,:), rho(:,:)
   real(kind_phys), allocatable :: z(:,:), pk(:,:)
   real(kind_phys), allocatable :: theta0(:,:), qv0(:,:), qc0(:,:), qr0(:,:)
   real(kind_phys), allocatable :: theta(:,:),  qv(:,:),  qc(:,:),  qr(:,:)
   real(kind_phys), allocatable :: precl(:), relhum(:,:)
   real(kind_phys), allocatable :: arr(:)

   ! ---------------------------------------------------------------------------
   ! RNG variables (Box-Muller, mirrors JD driver)
   ! ---------------------------------------------------------------------------
   integer,         allocatable :: seed_values(:)
   integer                      :: seed_size
   real(kind_phys)              :: u1, u2, ztmp

   ! ---------------------------------------------------------------------------
   ! Misc
   ! ---------------------------------------------------------------------------
   character(len=512) :: errmsg
   integer            :: errflg
   character(len=64)  :: scheme_name
   integer            :: i_ncol, ncol, i, k, irep

   integer, parameter :: IUNIT = 42
   character(len=256) :: outpath
   integer            :: io_stat

   outpath = '_compare_results/outputs/scalability/' // &
             'fortran_scalability_results.json'

   ! ---------------------------------------------------------------------------
   ! Initialize kessler module constants
   ! ---------------------------------------------------------------------------
   call kessler_init(LV_IN, PREF_IN, RHOQR_IN, errmsg, errflg)
   if (errflg /= 0) then
      write(*,*) 'ERROR: kessler_init failed: ', trim(errmsg)
      stop 1
   end if

   call system_clock(count_rate=cr)

   write(*,'(A)')       '======================================================='
   write(*,'(A)')       ' Kessler Fortran JD scalability benchmark'
   write(*,'(A)')       '======================================================='
   write(*,'(A,I0)')    ' System clock rate (Hz) : ', cr
   write(*,'(A,I0)')    ' nz                     : ', NZ
   write(*,'(A,F6.1)')  ' dt (s)                 : ', DT
   write(*,'(A,I0)')    ' N_REPS                 : ', N_REPS
   write(*,'(A)')       ' ncol sweep             : 50 100 200 500 1000 2000 5000 10000 100000 1000000'
   write(*,'(A)')       ' Input style            : JD physics-based (Box-Muller per-column scaling)'
   write(*,'(A)')       ' Backend                : CPU (serial, no parallelism)'
   write(*,*)

   ! ---------------------------------------------------------------------------
   ! Open JSON output
   ! ---------------------------------------------------------------------------
   open(unit=IUNIT, file=trim(outpath), status='replace', action='write', &
        iostat=io_stat)
   if (io_stat /= 0) then
      write(*,*) 'ERROR: cannot open output file: ', trim(outpath)
      write(*,*) '  Make sure the output directory exists:'
      write(*,*) '  mkdir -p _compare_results/outputs/scalability'
      stop 1
   end if

   write(IUNIT,'(A)') '{'
   write(IUNIT,'(A)') &
        '  "config": {"ncol_sweep": [50,100,200,500,1000,2000,5000,10000,100000,1000000],' // &
        ' "nz": 56, "dt": 60.0, "n_reps": 5, "backend": "cpu",' // &
        ' "input_style": "jd_physics_based"},'
   write(IUNIT,'(A)') '  "results": {'
   write(IUNIT,'(A)') '    "fortran": ['

   ! ---------------------------------------------------------------------------
   ! Main benchmark loop
   ! ---------------------------------------------------------------------------
   do i_ncol = 1, N_SWEEP
      ncol = NCOL_SWEEP(i_ncol)
      write(*,'(A,I0)') '--- ncol = ', ncol

      ! ---- Allocate ----
      allocate(arr(ncol))
      allocate(cpair(ncol,NZ), rair(ncol,NZ), rho(ncol,NZ), &
               z(ncol,NZ),     pk(ncol,NZ))
      allocate(theta0(ncol,NZ), qv0(ncol,NZ), qc0(ncol,NZ), qr0(ncol,NZ))
      allocate(theta(ncol,NZ),  qv(ncol,NZ),  qc(ncol,NZ),  qr(ncol,NZ))
      allocate(precl(ncol), relhum(ncol,NZ))

      ! ---- Per-column scaling via Box-Muller (mirrors JD_kessler_driver.F90) ----
      call random_seed(size=seed_size)
      allocate(seed_values(seed_size))
      seed_values = [(i, i=1, seed_size)]
      call random_seed(put=seed_values)
      deallocate(seed_values)

      do i = 1, ncol
         call random_number(u1)
         call random_number(u2)
         ztmp   = sqrt(-2.0_kind_phys * log(u1)) * cos(2.0_kind_phys * 3.14159265358979_kind_phys * u2)
         arr(i) = 1.0_kind_phys + 0.1_kind_phys * ztmp
      end do

      ! ---- Physics-based initialization (mirrors JD_kessler_driver.F90) ----
      do i = 1, ncol
         do k = 1, NZ
            cpair(i,k) = 1004.0_kind_phys
            rair(i,k)  = 287.0_kind_phys

            z(i,k)   = arr(i) * (100.0_kind_phys * real(k-1, kind_phys))
            rho(i,k) = arr(i) * (1.2_kind_phys * exp(-z(i,k) / 8000.0_kind_phys))
            pk(i,k)  = arr(i) * 1.0_kind_phys

            theta0(i,k) = arr(i) * (300.0_kind_phys - 0.006_kind_phys * z(i,k))
            qv0(i,k)    = arr(i) * 0.010_kind_phys
            qc0(i,k)    = arr(i) * 0.01_kind_phys
            qr0(i,k)    = arr(i) * 0.01_kind_phys
         end do
      end do

      ! ---- Warmup call ----
      theta = theta0;  qv = qv0;  qc = qc0;  qr = qr0
      precl = 0.0_kind_phys
      write(*,'(A)', advance='no') '  warmup...  '
      call system_clock(t_start)
      call kessler_run(ncol, NZ, DT, LYR_SURF, LYR_TOA, &
                       cpair, rair, rho, z, pk, &
                       theta, qv, qc, qr, precl, relhum, &
                       scheme_name, errmsg, errflg)
      call system_clock(t_end)
      warmup_s = real(t_end - t_start, 8) / real(cr, 8)

      if (errflg /= 0) then
         write(*,'(A,A)') 'FAILED: ', trim(errmsg)
         deallocate(arr, cpair, rair, rho, z, pk)
         deallocate(theta0, qv0, qc0, qr0, theta, qv, qc, qr)
         deallocate(precl, relhum)
         cycle
      end if
      write(*,'(A,F8.4,A)') 'done (', warmup_s, 's)'

      ! ---- Timed repetitions ----
      do irep = 1, N_REPS
         ! Re-initialize inout arrays before every call
         theta = theta0;  qv = qv0;  qc = qc0;  qr = qr0
         precl = 0.0_kind_phys
         call system_clock(t_start)
         call kessler_run(ncol, NZ, DT, LYR_SURF, LYR_TOA, &
                          cpair, rair, rho, z, pk, &
                          theta, qv, qc, qr, precl, relhum, &
                          scheme_name, errmsg, errflg)
         call system_clock(t_end)
         times(irep) = real(t_end - t_start, 8) / real(cr, 8)
      end do

      call compute_stats(times, N_REPS, median_s, mean_s, std_s)
      write(*,'(A,F10.4,A,F10.4,A)') &
           '  median=', median_s*1d3, ' ms   mean=', mean_s*1d3, ' ms'

      ! ---- Write JSON entry ----
      write(IUNIT,'(A)') '      {'
      write(IUNIT,'(A,I0,A)')     '        "ncol": ',    ncol,      ','
      write(IUNIT,'(A,ES15.8,A)') '        "warmup_s": ', warmup_s, ','
      write(IUNIT,'(A)', advance='no') '        "times_s": ['
      call write_array_json(IUNIT, times, N_REPS)
      write(IUNIT,'(A)') '],'
      write(IUNIT,'(A,ES15.8,A)') '        "median_s": ', median_s, ','
      write(IUNIT,'(A,ES15.8,A)') '        "mean_s": ',   mean_s,   ','
      write(IUNIT,'(A,ES15.8)')   '        "std_s": ',    std_s
      if (i_ncol < N_SWEEP) then
         write(IUNIT,'(A)') '      },'
      else
         write(IUNIT,'(A)') '      }'
      end if

      ! ---- Deallocate ----
      deallocate(arr, cpair, rair, rho, z, pk)
      deallocate(theta0, qv0, qc0, qr0, theta, qv, qc, qr)
      deallocate(precl, relhum)

   end do

   write(IUNIT,'(A)') '    ]'
   write(IUNIT,'(A)') '  }'
   write(IUNIT,'(A)') '}'
   close(IUNIT)

   write(*,*)
   write(*,'(A,A)') ' Results saved -> ', trim(outpath)

contains

   ! ---------------------------------------------------------------------------
   ! Compute median, mean, std of a double-precision array
   ! ---------------------------------------------------------------------------
   subroutine compute_stats(arr, n, median, mean, std)
      integer,     intent(in)  :: n
      real(kind=8),intent(in)  :: arr(n)
      real(kind=8),intent(out) :: median, mean, std
      real(kind=8) :: sorted(n), tmp, s, s2
      integer :: i, j

      sorted = arr
      ! Insertion sort (n=5, so trivially fast)
      do i = 2, n
         tmp = sorted(i)
         j   = i - 1
         do while (j >= 1 .and. sorted(j) > tmp)
            sorted(j+1) = sorted(j)
            j = j - 1
         end do
         sorted(j+1) = tmp
      end do

      if (mod(n, 2) == 1) then
         median = sorted(n/2 + 1)
      else
         median = (sorted(n/2) + sorted(n/2 + 1)) * 0.5d0
      end if

      s = 0.0d0
      do i = 1, n
         s = s + arr(i)
      end do
      mean = s / real(n, 8)

      s2 = 0.0d0
      do i = 1, n
         s2 = s2 + (arr(i) - mean)**2
      end do
      std = sqrt(s2 / real(n, 8))

   end subroutine compute_stats

   ! ---------------------------------------------------------------------------
   ! Write a real(8) array as comma-separated JSON values (no brackets)
   ! ---------------------------------------------------------------------------
   subroutine write_array_json(iunit, arr, n)
      integer,     intent(in) :: iunit, n
      real(kind=8),intent(in) :: arr(n)
      integer :: i
      do i = 1, n
         if (i < n) then
            write(iunit,'(ES15.8,A)', advance='no') arr(i), ', '
         else
            write(iunit,'(ES15.8)',   advance='no') arr(i)
         end if
      end do
   end subroutine write_array_json

end program scalability_benchmark_fortran
