program test_kessler_driver

  use omp_lib
  use iso_c_binding
  use kessler
  use kessler_update

  implicit none

  integer, parameter :: kind_phys = selected_real_kind(12)

  ! Dimensions
  integer :: ncol, nz
  integer :: lyr_surf, lyr_toa
  real(kind_phys) :: dt

  ! Init constants
  real(kind_phys) :: lv_in, pref_in, rhoqr_in

  ! Arrays
  real(kind_phys), allocatable :: cpair(:,:), rair(:,:), rho(:,:)
  real(kind_phys), allocatable :: z(:,:), pk(:,:)
  real(kind_phys), allocatable :: theta(:,:), qv(:,:), qc(:,:), qr(:,:)
  real(kind_phys), dimension(:,:), allocatable :: temp, exner, zm, temp_prev, st_energy, ttend_t
  real(kind_phys), allocatable :: precl(:), phis(:)
  real(kind_phys), allocatable :: relhum(:,:)
  real(kind_phys), allocatable :: arr(:)
  real(kind_phys) :: u1, u2, ztmp

  character(len=64)  :: scheme_name
  character(len=512) :: errmsg
  integer            :: errflg
  integer, allocatable :: seed_values(:)
  integer :: seed_size

  logical :: use_host
  integer :: dev
  integer(c_size_t) :: bytes1D, bytes2D
  type(c_ptr) :: cpair_raw, rair_raw, rho_raw, z_raw, pk_raw, theta_raw, &
                 qv_raw, qc_raw, qr_raw, precl_raw, relhum_raw

  type(c_ptr) :: temp_raw, exner_raw, zm_raw, phis_raw, temp_prev_raw, &
                 st_energy_raw, ttend_t_raw

  integer :: i, k, ierr

  !------------------------------------------------------
  ! Set grid size
  !------------------------------------------------------
  !ncol = 2
  ncol = 128
  nz   = 56
  dt   = 60.0_kind_phys

  lyr_surf = 1
  lyr_toa  = nz

  !------------------------------------------------------
  ! Initialize Kessler constants
  !------------------------------------------------------
  lv_in    = 2.5e6_kind_phys     ! J/kg
  pref_in  = 100000.0_kind_phys  ! Pa (1000 hPa)
  rhoqr_in = 1000.0_kind_phys    ! kg/m^3

  call kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg)

  if (errflg /= 0) then
     print *, 'Initialization error: ', trim(errmsg)
     stop
  end if

  !------------------------------------------------------
  ! Allocate arrays
  !------------------------------------------------------
  allocate(cpair(ncol,nz), rair(ncol,nz), rho(ncol,nz))
  allocate(z(ncol,nz), pk(ncol,nz))
  allocate(theta(ncol,nz), qv(ncol,nz), qc(ncol,nz), qr(ncol,nz))
  allocate(relhum(ncol,nz))
  allocate(temp(ncol,nz))
  allocate(exner(ncol,nz))
  allocate(zm(ncol,nz))
  allocate(temp_prev(ncol,nz))
  allocate(st_energy(ncol,nz))
  allocate(ttend_t(ncol,nz))

  allocate(precl(ncol))
  allocate(phis(ncol))

  allocate(arr(ncol))
  ! Query the size of the RNG state
  call random_seed(size=seed_size)
  allocate(seed_values(seed_size))

  ! Fill with fixed values for deterministic sequence
  seed_values = [(i, i=1, seed_size)]  ! or any fixed sequence

  ! Set the seed
  call random_seed(put=seed_values)

  do i = 1, ncol
     ! Box-Muller transform to generate standard normal
     call random_number(u1)
     call random_number(u2)
     ztmp = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265 * u2)

     ! Scale to mean=1, stddev=0.1
     arr(i) = 1.0 + 0.1 * ztmp
  end do
  !------------------------------------------------------
  ! Simple initialization
  !------------------------------------------------------
  do i = 1, ncol
     do k = 1, nz

        cpair(i,k) = 1004.0_kind_phys
        rair(i,k)  = 287.0_kind_phys

        z(i,k)   = arr(i) * (100.0_kind_phys * real(k-1, kind_phys))
        rho(i,k) = arr(i) * (1.2_kind_phys * exp(-z(i,k)/8000.0_kind_phys))

        pk(i,k)    = arr(i) * (1.0_kind_phys)
        theta(i,k) = arr(i) * (300.0_kind_phys - 0.006_kind_phys*z(i,k))

        qv(i,k) = arr(i) * (0.010_kind_phys)
        qc(i,k) = arr(i) * (0.01_kind_phys)
        qr(i,k) = arr(i) * (0.01_kind_phys)

        temp(i,k)  = arr(i) * 287.4_kind_phys
        zm(i,k)    = zm(i,k)
        exner(i,k) = arr(i) * 0.97_kind_phys
     end do
     phis(i) = arr(i) * 0.1_kind_phys  ! approximately over ocean
  end do

#ifdef USE_GPU
  dev = omp_get_default_device()


  bytes2D = ncol * nz * c_sizeof(z(1,1))
  bytes1D = ncol * c_sizeof(precl(1))

  ! Allocate 2D device memory
  cpair_raw  = omp_target_alloc(bytes2D, dev)
  rair_raw   = omp_target_alloc(bytes2D, dev)
  rho_raw    = omp_target_alloc(bytes2D, dev)
  z_raw      = omp_target_alloc(bytes2D, dev)
  pk_raw     = omp_target_alloc(bytes2D, dev)
  theta_raw  = omp_target_alloc(bytes2D, dev)
  qv_raw     = omp_target_alloc(bytes2D, dev)
  qc_raw     = omp_target_alloc(bytes2D, dev)
  qr_raw     = omp_target_alloc(bytes2D, dev)
  relhum_raw = omp_target_alloc(bytes2D, dev)

  temp_raw       = omp_target_alloc(bytes2D, dev)
  exner_raw      = omp_target_alloc(bytes2D, dev)
  zm_raw         = omp_target_alloc(bytes2D, dev)
  temp_prev_raw  = omp_target_alloc(bytes2D, dev)
  st_energy_raw  = omp_target_alloc(bytes2D, dev)
  ttend_t_raw    = omp_target_alloc(bytes2D, dev)

  ! Associate 2D device arrays to raw pointers
  ierr = omp_target_associate_ptr(c_loc(cpair(1,1)), cpair_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(rair(1,1)), rair_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(rho(1,1)), rho_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(z(1,1)), z_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(pk(1,1)), pk_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(theta(1,1)), theta_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(qv(1,1)), qv_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(qc(1,1)), qc_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(qr(1,1)), qr_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(relhum(1,1)), relhum_raw, bytes2D, 0_c_size_t, dev)

  ierr = omp_target_associate_ptr(c_loc(temp(1,1)), temp_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(exner(1,1)), exner_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(zm(1,1)), zm_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(temp_prev(1,1)), temp_prev_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(st_energy(1,1)), st_energy_raw, bytes2D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(ttend_t(1,1)), ttend_t_raw, bytes2D, 0_c_size_t, dev)

  ! Allocate 1D device memory
  precl_raw  = omp_target_alloc(bytes1D, dev)
  phis_raw   = omp_target_alloc(bytes1D, dev)

  ! Associate 1D device arrays to raw pointers
  ierr = omp_target_associate_ptr(c_loc(precl(1)), precl_raw, bytes1D, 0_c_size_t, dev)
  ierr = omp_target_associate_ptr(c_loc(phis(1)), phis_raw, bytes1D, 0_c_size_t, dev)

  ! Host -> device memcpy
  !$omp target update to(cpair(1:ncol,1:nz), rair(1:ncol,1:nz), rho(1:ncol,1:nz), &
  !$omp      z(1:ncol,1:nz), pk(1:ncol,1:nz), theta(1:ncol,1:nz), qv(1:ncol,1:nz), &
  !$omp      qc(1:ncol,1:nz), qr(1:ncol,1:nz), precl(1:ncol),relhum(1:ncol,1:nz))

#endif

  !------------------------------------------------------
  ! Run microphysics
  !------------------------------------------------------
  call kessler_run(ncol, nz, dt, lyr_surf, lyr_toa, &
                   cpair, rair, rho, z, pk, &
                   theta, qv, qc, qr, &
                   precl, relhum, scheme_name, errmsg, errflg)

#ifdef USE_GPU
  ! Device -> host  memcpy
  !$omp target update from(theta(1:ncol,1:nz),qv(1:ncol,1:nz), &
  !$omp      qc(1:ncol,1:nz),qr(1:ncol,1:nz),precl(1:ncol),relhum(1:ncol,1:nz))
#endif

  !------------------------------
  ! kessler_update_timestep_init
  !------------------------------

#ifdef USE_GPU
  !$omp target update to(temp(1:ncol,1:nz))
#endif
  
  call kessler_update_timestep_init(temp, temp_prev, ttend_t, errmsg, errflg)

#ifdef USE_GPU
  !$omp target update from(temp_prev(1:ncol,1:nz),ttend_t(1:ncol,1:nz))
#endif

  !-------------------
  ! Kessler_update_run
  !-------------------
#ifdef USE_GPU
  !$omp target update to(theta(1:ncol,1:nz), exner(1:ncol,1:nz), &
  !$omp      temp_prev(1:ncol,1:nz), ttend_t(1:ncol,1:nz))
#endif

  call kessler_update_run(nz, ncol, dt, theta, exner, temp_prev, &
             ttend_t, errmsg, errflg)

#ifdef USE_GPU
  !$omp target update from(ttend_t(1:ncol,1:nz))
#endif

  !------------------------------
  ! Kessler_update_timestep_final
  !------------------------------
#ifdef USE_GPU
  !$omp target update to(cpair(1:ncol,1:nz), temp(1:ncol,1:nz), &
  !$omp      zm(1:ncol,1:nz),phis(1:ncol))
#endif

   call kessler_update_timestep_final(nz, cpair, temp, zm, phis, &
              st_energy, errflg, errmsg)

#ifdef USE_GPU
   !$omp target update from(st_energy(1:ncol,1:nz))
#endif


  if (errflg /= 0) then
     print *, 'Run error: ', trim(errmsg)
  else
     print *, 'Scheme name: ', trim(scheme_name)
     print *, 'theta: ', SUM(theta)
     print *, 'qv: ', SUM(qv)
     print *, 'qc: ', SUM(qc)
     print *, 'qr: ', SUM(qr)
     print *, 'Precip (m/s): ', SUM(precl)
     print *, 'relnum: ', SUM(relhum)
     print *, 'temp_prev: ', SUM(temp_prev)
     print *, 'ttend_t: ', SUM(ttend_t)
     print *, 'st_energy: ', SUM(st_energy)
  end if

end program test_kessler_driver
