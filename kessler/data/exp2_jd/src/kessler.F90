module kessler

   use ccpp_kinds, only:  kind_phys

   implicit none
   private
   save

   public :: kessler_init ! init routine
   public :: kessler_run  ! main routine

   ! Private module data (constants set at initialization)
   real(kind_phys) :: lv    ! latent heat of vaporization, J/kg
   real(kind_phys) :: pref  ! reference pressure, hPa
   real(kind_phys) :: rhoqr ! density of fresh liquid water, kg/m^3

CONTAINS

   !> \section arg_table_kessler_init  Argument Table
   !! \htmlinclude kessler_init.html
   subroutine kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg)
      ! Set physical constants to be consistent with calling model
      real(kind_phys),    intent(in)  :: lv_in    ! latent heat of vaporization, J/kg
      real(kind_phys),    intent(in)  :: pref_in  ! reference pressure, Pa
      real(kind_phys),    intent(in)  :: rhoqr_in ! density of fresh liquid water, kg/m^3

      character(len=512), intent(out) :: errmsg
      integer,            intent(out) :: errflg

      errmsg = ''
      errflg = 0

      lv    = lv_in
      pref  = pref_in/100._kind_phys
      rhoqr = rhoqr_in

   end subroutine kessler_init

   !-----------------------------------------------------------------------
   !
   !  Date:  June 26th, 2020
   !
   !  Change log:
   !  The Kessler warm rain scheme was first included to support the Dynamical Core
   !  Model Intercomparison Project (DCMIP) in 2016.
   !  in 2020:  Reformulation of the sub-cycling of the moisture processes to obey
   !  CFL condition for the sedimentation process. Corrected 1/kappa constant.
   !  Improved inline documentation of the Kessler processes.
   !
   !  The KESSLER subroutine implements the Kessler (1969) microphysics
   !  parameterization as described by Soong and Ogura (1973) and Klemp
   !  and Wilhelmson (1978). KESSLER is called at the end of each
   !  time step and makes the final adjustments to the potential
   !  temperature and moisture variables due to microphysical processes
   !  occurring during that time step. KESSLER is called once for each
   !  vertical column of grid cells. Increments are computed and added
   !  into the respective variables. The Kessler scheme contains three
   !  moisture categories: water vapor, cloud water (liquid water that
   !  moves with the flow), and rain water (liquid water that falls
   !  relative to the surrounding air). There  are no ice categories.
   !  The vertical loops in the column are ordered from the surface to the top.
   !
   !  Authors: Christiane Jablonowski (cjablono@umich.edu)
   !           University of Michigan, Ann Arbor
   !
   !           Paul Ullrich (paullrich@ucdavis.edu)
   !           University of California, Davis
   !
   !           Based on a code by Joseph Klemp
   !           (National Center for Atmospheric Research)
   !
   !  References:
   !
   !    Klemp J, B., and R. B. Wilhelmson, 1978: The Simulation of Three-
   !    Dimensional Convective Storm Dynamics. Journal of the Atmospheric
   !    Sciences, Vol. 35, 1070-1096
   !
   !    Durran, D. R., and J. B. Klemp, 1983: A Compressible Model for the
   !    Simulation of Moist Mountain Waves. Monthly Weather Review, Vol. 111,
   !    2341-2361
   !
   !    Klemp, J. B., W. C. Skamarock, W. C., and S.-H. Park, 2015:
   !    Idealized Global Nonhydrostatic Atmospheric Test Cases on a Reduced
   !    Radius Sphere. Journal of Advances in Modeling Earth Systems,
   !    Vol. 7, 1155-1177, doi:10.1002/2015MS000435
   !
   !=======================================================================

   !> \section arg_table_kessler_run  Argument Table
   !! \htmlinclude kessler_run.html
   subroutine kessler_run(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, &
        pk, theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg)

      !------------------------------------------------
      !   Input / output parameters
      !------------------------------------------------
      integer,          intent(in)    :: ncol       ! Number of columns
      integer,          intent(in)    :: nz         ! Number of vertical levels
      real(kind_phys),  intent(in)    :: dt         ! Physics time step (s)
      integer,          intent(in)    :: lyr_surf   ! Index of surface layer in the vertical coordinate
      integer,          intent(in)    :: lyr_toa    ! Index of top of the atmosphere in the vertical coordinate
      real(kind_phys),  intent(in)    :: cpair(:,:) ! Specific_heat_of_dry_air_at_constant_pressure (J/kg/K)
      real(kind_phys),  intent(in)    :: rair(:,:)  ! Gas constant of dry air (J/kg/K)
      real(kind_phys),  intent(in)    :: rho(:,:)   ! Dry air density (kg/m^3)
      real(kind_phys),  intent(in)    :: z(:,:)     ! Heights of thermo. levels (m)
      real(kind_phys),  intent(in)    :: pk(:,:)    ! Exner function (p/p0)**(R/cp)

      real(kind_phys),  intent(inout) :: theta(:,:) ! Potential temperature (K)
      real(kind_phys),  intent(inout) :: qv(:,:)    ! Water vapor mixing ratio wrt dry air (kg/kg)
      real(kind_phys),  intent(inout) :: qc(:,:)    ! Cloud water mixing ratio wrt dry air (kg/kg)
      real(kind_phys),  intent(inout) :: qr(:,:)    ! Rain water mixing ratio wrt dry air (kg/kg)

      real(kind_phys),  intent(out)   :: precl(:)   ! Precipitation rate (m_water / s)

      real(kind_phys),  intent(out)   :: relhum(:,:)! Relative humidity in percent

      character(len=64),intent(out)   :: scheme_name
      character(len=*), intent(out)   :: errmsg
      integer,          intent(out)   :: errflg

      !------------------------------------------------
      !   Local variables
      !------------------------------------------------
      real(kind_phys) :: r(ncol,nz),         &          ! Density in gm/(cm)^3
                         rhalf(ncol,nz),     &          ! sqrt ( density (lowest_model_level) / density (model_level))
                         velqr(ncol,nz),     &          ! Terminal fall speed of rain water (m/s)
                         sed(ncol,nz),       &          ! Sedimentation rate
                         pc(ncol,nz)                   ! Parameter: 3.8 hPa / pressure (in hPa)

      real(kind_phys) :: f5(ncol,nz),   &          ! Parameter for the computation of the condensation rate
                         f2x,           &          ! Parameter for the computation of the saturation mixing ratio
                         xk,            &          ! 1/kappa = cp/R
                         ern,           &          ! Evaporization rate of rain water
                         qrprod,        &          ! qc & qr changes due to autoconversion and collection of cloud water by rain
                         prod,          &          ! Used to compute condensation rate
                         qvs,           &          ! Saturation mixing ratio (in gm/gm)
                         dt0(ncol),     &          ! Subcycling time step (obeys 80% of the CFL constraint in the vertical)
                         mask(ncol)                ! Mask out columns that have already converged

      real(kind_phys) :: time_counter(ncol),  &    ! Elapsed time during the subcycling steps
                         precl_acc(ncol)           ! Time-weighted accumulation of the precipitation rate

      integer         :: col, klev                 ! Column and level indices
      integer         :: lyr_step                  ! Increment to move up a level
      integer         :: colError
      integer         :: iterCnt
      logical         :: all_converged
      real(kind_phys) :: dtmin

      ! Initialize output variables
      precl = 0._kind_phys
      errmsg = ''
      errflg = 0
      scheme_name = "KESSLER"

      ! Check inputs
      if (dt <= 0._kind_phys) then
         write(errmsg,*) 'KESSLER called with nonpositive dt'
         errflg = 1
         return
      end if

      if (lyr_surf > lyr_toa) then
         lyr_step = -1
      else
         lyr_step = 1
      end if

      f2x = 17.27_kind_phys  ! constant for the saturation mixing ratio

      !$acc data create(r,rhalf,velqr,sed,pc,f5,dt0,mask,time_counter,precl_acc)
      !------------------------------------------------
      !   Begin calculation
      !------------------------------------------------
      !!$acc parallel deviceptr(cpair,rair,pk,rho,qr,z)
      !$acc parallel

      ! Loop through columns
      !$acc loop collapse(2)
      do col =1,ncol
         do klev=lyr_surf, lyr_toa, lyr_step

            !Calculate constants:
            f5(col,klev)  = 4093._kind_phys * lv / cpair(col,klev) ! constant for the condensation rate
            xk  = cpair(col,klev) / rair(col,klev)       ! 1/kappa = cp/R

            r(col,klev)     = 0.001_kind_phys * rho(col, klev)
            rhalf(col,klev) = sqrt(rho(col, lyr_surf) / rho(col, klev))
            pc(col,klev)    = 3.8_kind_phys / ((pk(col, klev)**xk) * pref)
            !
            ! if qr is (round-off) negative then the computation of
            ! velqr triggers floating point exception error when running
            ! in debugging mode with NAG
            !
            qr(col,klev) = MAX(qr(col,klev),0.0_kind_phys)
            !
            ! Liquid water terminal velocity (m/s) following Klemp and Wilhelmson (1978), Eq. (2.15)
            velqr(col,klev)  = 36.34_kind_phys * rhalf(col,klev) *          &
                 (qr(col, klev) * r(col,klev))**0.1364_kind_phys
        enddo
      enddo

      ! Compute maximum time step size in accordance with CFL condition
      !$acc loop private(dtmin)
      do col=1,ncol
         dt0(col)  = dt
         mask(col) = 1.0
         dtmin = dt0(col)
         !$acc loop reduction(min:dtmin)
         do  klev=lyr_surf,lyr_toa - lyr_step,lyr_step
           ! NB: Original test for velqr /= 0 numerically unstable
           if (abs(velqr(col,klev)) > 1.0E-12_kind_phys) then
              dtmin = min(dtmin, 0.8_kind_phys*(z(col, klev+lyr_step) - &
                  z(col, klev)) / velqr(col,klev))
           end if
         enddo
         dt0(col) = dtmin
         ! time counter keeps track of the elapsed time during the subcycling process
         time_counter(col) = 0.0_kind_phys

         ! initialize time-weighted accumulated precipitation
         precl_acc(col) = 0.0_kind_phys
      enddo

      !$acc loop
      do col = 1, ncol
         ! Check the time step dt0
         if (dt0(col) <  1.0E-12_kind_phys) then
            !$acc atomic write
            colError = col
            !$acc atomic write
            errflg = 1
         end if
      enddo
      !$acc end parallel

      if(errflg .eq. 1) then 
         write(errmsg, *) 'KESSLER: bad time splitting ',dt,dt0(colError)
         return
       endif

      iterCnt = 0
      all_converged = .FALSE.
      ! Subcycle through the Kessler moisture processes,
      ! time loop ends when the physics time step is reached (within a margin of 1e-5 s)
      ! do while ( abs(dt - time_counter(col)) > 1.0E-5_kind_phys)
      do while ( .not. all_converged)

         !!$acc parallel deviceptr(qr,qc,qv,pk,z,theta,rho,precl,cpair)
         !$acc parallel
         !$acc loop
         do col = 1, ncol
            ! Precipitation rate (m_water/s) over the subcycled time step
            precl(col) = rho(col, lyr_surf) * qr(col, lyr_surf) * velqr(col,lyr_surf) / rhoqr

            ! accumulate the preciptation rate over the subcycled time steps
            ! (weighted with the subcycled time step), unit is m_water
            precl_acc(col) = precl_acc(col) + mask(col)*(precl(col) * dt0(col))
         enddo

         ! Mass-weighted sedimentation term using upstream differencing
         !$acc loop
         do col = 1, ncol
            do klev = lyr_surf, lyr_toa - lyr_step, lyr_step
               sed(col,klev) = dt0(col) *                                                           &
                    ((r(col,klev+lyr_step) * qr(col, klev+lyr_step) * velqr(col,klev+lyr_step)) -  &
                     (r(col,klev) * qr(col, klev) * velqr(col,klev))) /                            &
                    (r(col,klev) * (z(col, klev+lyr_step) - z(col, klev)))
            end do
         enddo

         !$acc loop
         do col = 1, ncol
            sed(col,lyr_toa) = -dt0(col) * qr(col, lyr_toa) * velqr(col,lyr_toa) /    &
                 (0.5_kind_phys * (z(col, lyr_toa)-z(col, lyr_toa-lyr_step)))
         enddo

         !$acc loop collapse(2)
         do col = 1, ncol
            ! Adjustment terms
            do klev = lyr_surf, lyr_toa, lyr_step

               ! Autoconversion and collection rates following Klemp and Wilhelmson (1978), Eqs. (2.13a,b)
               ! the collection process is handled with a semi-implicit time stepping approach
               qrprod = qc(col, klev) - (qc(col, klev) - dt0(col) *           &
                    max(.001_kind_phys * (qc(col, klev)-.001_kind_phys), &
                        0._kind_phys)) /                                 &
                        (1._kind_phys + dt0(col) * 2.2_kind_phys *            &
                         qr(col, klev)**.875_kind_phys)
               qc(col, klev) = max(qc(col, klev) - qrprod, 0._kind_phys)
               qr(col, klev) = max(qr(col, klev) + qrprod + sed(col,klev), 0._kind_phys)

               ! Teten's formula: saturation vapor mixing ratio (gm/gm) following Klemp and Wilhelmson (1978), Eq. (2.11)
               qvs = pc(col,klev) * exp(f2x*(pk(col, klev)*theta(col, klev) - 273._kind_phys) / (pk(col, klev)*theta(col, klev) &
                              - 36._kind_phys))
               ! Temporary variable for the condensation rate, following Durran and Klemp (1983), Eqs. (A13-A14)
               prod = (qv(col, klev) - qvs) / (1._kind_phys + qvs*f5(col,klev) / (pk(col, klev)*theta(col, klev) - 36._kind_phys)**2)

               ! Evaporation rate following Klemp and Wilhelmson (1978) Eq. (2.14a,b), also Durran and Klemp (1983) Eqs. (A8-A9)
               ern = min(dt0(col) * (((1.6_kind_phys + 124.9_kind_phys*(r(col,klev)*qr(col, klev))**.2046_kind_phys) * &
                    (r(col,klev) * qr(col, klev))**.525_kind_phys) /                                              &
                    (2550000._kind_phys * pc(col,klev) / (3.8_kind_phys*qvs) + 540000._kind_phys)) *              &
                    (dim(qvs,qv(col, klev)) / (r(col,klev)*qvs)),                                                 &
                    max(-prod-qc(col, klev),0._kind_phys),qr(col, klev))

               ! Saturation adjustment following Durran and Klemp (1983) Eqs. (A1-A4), also Klemp and Wilhelmson (1978) Eq. (3.10)
               theta(col, klev)= theta(col, klev) + mask(col) * (lv / (cpair(col,klev) * pk(col, klev)) * (max(prod,-qc(col, klev)) - ern))
               qv(col, klev) = mask(col) * (max(qv(col, klev) - max(prod, -qc(col, klev)) + ern, 0._kind_phys)) &
                             + (1._kind_phys - mask(col))*qv(col,klev)
               qc(col, klev) = qc(col, klev) + mask(col) * max(prod, -qc(col, klev))
               qr(col, klev) = mask(col) * (max(qr(col, klev) - ern, 0._kind_phys)) &
                             + (1._kind_phys - mask(col))*qr(col, klev)
            end do
          enddo

          !$acc loop
          do col = 1, ncol
          ! Compute the elapsed time
            time_counter(col) = time_counter(col) + mask(col) * dt0(col)
            dt0(col) = max(dt -  time_counter(col), 0.0_kind_phys)
            ! Construct a mask that if a columns have satisfied an exit condition
            !    has not converged = 1.0
            !    has converged = 0.0
             if (abs(dt - time_counter(col)) > 1.0E-5_kind_phys) then 
                mask(col) = 1._kind_phys
             else
                mask(col) = 0._kind_phys
             endif
          end do ! column loop

          ! Recalculate liquid water terminal velocity (m/s)
          !$acc loop collapse(2)
          do col = 1, ncol
             do klev = lyr_surf, lyr_toa, lyr_step
                velqr(col,klev)  = 36.34_kind_phys * rhalf(col,klev) * (qr(col, klev)*r(col,klev))**0.1364_kind_phys
             end do
          end do ! column loop

          ! recompute the time step
          !$acc loop private(dtmin)
          do col = 1, ncol
             dtmin = dt0(col)
             !$acc loop reduction(min:dtmin)
             do klev = lyr_surf, lyr_toa - lyr_step, lyr_step
                if (abs(velqr(col,klev)) > 1.0E-12_kind_phys) then
                   dtmin = min(dtmin, 0.8_kind_phys*(z(col, klev+lyr_step) - z(col, klev)) / velqr(col,klev))
                end if
             end do
             dt0(col) = dtmin
          end do ! column loop
          !$acc end parallel

          ! check to see if all columns have satisfied the condition
          all_converged = all_equal(mask,0._kind_phys)
          iterCnt=iterCnt+1

      end do  ! do while loop

      !!$acc parallel deviceptr(pk,theta,relhum,qv,precl)
      !$acc parallel

      !$acc loop
      do col=1,ncol
         ! compute the average preciptation rate over the physics time step period
         precl(col) = precl_acc(col) / dt
      end do ! column loop

      ! Diagnostic: relative humidity (relhum)
      !$acc loop collapse(2)
      do col = 1,ncol
         do klev = lyr_surf,lyr_toa,lyr_step
            ! Saturation vapor mixing ratio (gm/gm)
            qvs = pc(col,klev) * exp(f2x*(pk(col, klev)*theta(col, klev) - 273._kind_phys) / (pk(col, klev)*theta(col, klev) &
                           - 36._kind_phys))
            relhum(col,klev) = qv(col,klev) / qvs * 100._kind_phys ! in percent
          enddo
      end do ! column loop
      !$acc end parallel

      !$acc end data
      print *,'iteration Cnt: ',iterCnt

   end subroutine kessler_run

   !=======================================================================
   logical function all_equal(data,val) 

      implicit none
      real(kind_phys), intent(in) :: data(:)
      real(kind_phys), intent(in) :: val

      ! local variables
      integer :: i, n
      logical :: result

      result = .true.

      n = size(data)

      !$acc parallel loop reduction(.and.:result)
      do i= 1, n
         if(data(i) .ne. val) then 
             result = .false.
         endif
      enddo
      all_equal = result

    end function all_equal

end module kessler
