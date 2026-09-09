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
      real(kind_phys) :: r(nz),         &          ! Density in gm/(cm)^3
                         rhalf(nz),     &          ! sqrt ( density (lowest_model_level) / density (model_level))
                         velqr(nz),     &          ! Terminal fall speed of rain water (m/s)
                         sed(nz),       &          ! Sedimentation rate
                         pc(nz)                    ! Parameter: 3.8 hPa / pressure (in hPa)

      real(kind_phys) :: f5,            &          ! Parameter for the computation of the condensation rate
                         f2x,           &          ! Parameter for the computation of the saturation mixing ratio
                         xk,            &          ! 1/kappa = cp/R
                         ern,           &          ! Evaporization rate of rain water
                         qrprod,        &          ! qc & qr changes due to autoconversion and collection of cloud water by rain
                         prod,          &          ! Used to compute condensation rate
                         qvs,           &          ! Saturation mixing ratio (in gm/gm)
                         dt0                       ! Subcycling time step (obeys 80% of the CFL constraint in the vertical)

      real(kind_phys) :: time_counter,  &          ! Elapsed time during the subcycling steps
                         precl_acc                 ! Time-weighted accumulation of the precipitation rate

      integer         :: col, klev                 ! Column and level indices
      integer         :: lyr_step                  ! Increment to move up a level

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

      !------------------------------------------------
      !   Begin calculation
      !------------------------------------------------

      ! Loop through columns
      do col = 1, ncol
         do klev = lyr_surf, lyr_toa, lyr_step

            !Calculate constants:
            f5  = 4093._kind_phys * lv / cpair(col,klev) ! constant for the condensation rate
            xk  = cpair(col,klev) / rair(col,klev)       ! 1/kappa = cp/R

            r(klev)     = 0.001_kind_phys * rho(col, klev)
            rhalf(klev) = sqrt(rho(col, lyr_surf) / rho(col, klev))
            pc(klev)    = 3.8_kind_phys / ((pk(col, klev)**xk) * pref)
            !
            ! if qr is (round-off) negative then the computation of
            ! velqr triggers floating point exception error when running
            ! in debugging mode with NAG
            !
            qr(col,klev) = MAX(qr(col,klev),0.0_kind_phys)
            !
            ! Liquid water terminal velocity (m/s) following Klemp and Wilhelmson (1978), Eq. (2.15)
            velqr(klev)  = 36.34_kind_phys * rhalf(klev) *          &
                 (qr(col, klev) * r(klev))**0.1364_kind_phys
         end do

         ! Compute maximum time step size in accordance with CFL condition
         dt0 = dt
         do klev = lyr_surf, lyr_toa - lyr_step, lyr_step
            ! NB: Original test for velqr /= 0 numerically unstable
            if (abs(velqr(klev)) > 1.0E-12_kind_phys) then
               dt0 = min(dt0, 0.8_kind_phys*(z(col, klev+lyr_step) - &
                    z(col, klev)) / velqr(klev))
            end if
         end do

         ! Check the time step dt0
         if (dt0 <  1.0E-12_kind_phys) then
            write(errmsg, *) 'KESSLER: bad time splitting ',dt,dt0
            errflg = 1
            return
         end if

         ! time counter keeps track of the elapsed time during the subcycling process
         time_counter = 0.0_kind_phys

         ! initialize time-weighted accumulated precipitation
         precl_acc = 0.0_kind_phys
         ! Subcycle through the Kessler moisture processes,
         ! time loop ends when the physics time step is reached (within a margin of 1e-5 s)
         do while ( abs(dt - time_counter) > 1.0E-5_kind_phys)

            ! Precipitation rate (m_water/s) over the subcycled time step
            precl(col) = rho(col, lyr_surf) * qr(col, lyr_surf) * velqr(lyr_surf) / rhoqr

            ! accumulate the preciptation rate over the subcycled time steps
            ! (weighted with the subcycled time step), unit is m_water
            precl_acc = precl_acc + precl(col) * dt0

            ! Mass-weighted sedimentation term using upstream differencing
            do klev = lyr_surf, lyr_toa - lyr_step, lyr_step
               sed(klev) = dt0 *                                                           &
                    ((r(klev+lyr_step) * qr(col, klev+lyr_step) * velqr(klev+lyr_step)) -  &
                     (r(klev) * qr(col, klev) * velqr(klev))) /                            &
                    (r(klev) * (z(col, klev+lyr_step) - z(col, klev)))
            end do
            sed(lyr_toa) = -dt0 * qr(col, lyr_toa) * velqr(lyr_toa) /    &
                 (0.5_kind_phys * (z(col, lyr_toa)-z(col, lyr_toa-lyr_step)))

            ! Adjustment terms
            do klev = lyr_surf, lyr_toa, lyr_step

               ! Autoconversion and collection rates following Klemp and Wilhelmson (1978), Eqs. (2.13a,b)
               ! the collection process is handled with a semi-implicit time stepping approach
               qrprod = qc(col, klev) - (qc(col, klev) - dt0 *           &
                    max(.001_kind_phys * (qc(col, klev)-.001_kind_phys), &
                        0._kind_phys)) /                                 &
                        (1._kind_phys + dt0 * 2.2_kind_phys *            &
                         qr(col, klev)**.875_kind_phys)
               qc(col, klev) = max(qc(col, klev) - qrprod, 0._kind_phys)
               qr(col, klev) = max(qr(col, klev) + qrprod + sed(klev), 0._kind_phys)

               ! Teten's formula: saturation vapor mixing ratio (gm/gm) following Klemp and Wilhelmson (1978), Eq. (2.11)
               qvs = pc(klev) * exp(f2x*(pk(col, klev)*theta(col, klev) - 273._kind_phys) / (pk(col, klev)*theta(col, klev) &
                              - 36._kind_phys))
               ! Temporary variable for the condensation rate, following Durran and Klemp (1983), Eqs. (A13-A14)
               prod = (qv(col, klev) - qvs) / (1._kind_phys + qvs*f5 / (pk(col, klev)*theta(col, klev) - 36._kind_phys)**2)

               ! Evaporation rate following Klemp and Wilhelmson (1978) Eq. (2.14a,b), also Durran and Klemp (1983) Eqs. (A8-A9)
               ern = min(dt0 * (((1.6_kind_phys + 124.9_kind_phys*(r(klev)*qr(col, klev))**.2046_kind_phys) * &
                    (r(klev) * qr(col, klev))**.525_kind_phys) /                                              &
                    (2550000._kind_phys * pc(klev) / (3.8_kind_phys*qvs) + 540000._kind_phys)) *              &
                    (dim(qvs,qv(col, klev)) / (r(klev)*qvs)),                                                 &
                    max(-prod-qc(col, klev),0._kind_phys),qr(col, klev))

               ! Saturation adjustment following Durran and Klemp (1983) Eqs. (A1-A4), also Klemp and Wilhelmson (1978) Eq. (3.10)
               theta(col, klev)= theta(col, klev) + (lv / (cpair(col,klev) * pk(col, klev)) * (max(prod,-qc(col, klev)) - ern))
               qv(col, klev) = max(qv(col, klev) - max(prod, -qc(col, klev)) + ern, 0._kind_phys)
               qc(col, klev) = qc(col, klev) + max(prod, -qc(col, klev))
               qr(col, klev) = max(qr(col, klev) - ern, 0._kind_phys)
            end do

          ! Compute the elapsed time
            time_counter = time_counter + dt0

          ! Recalculate liquid water terminal velocity (m/s)
             do klev = lyr_surf, lyr_toa, lyr_step
                velqr(klev)  = 36.34_kind_phys * rhalf(klev) * (qr(col, klev)*r(klev))**0.1364_kind_phys
             end do

          ! recompute the time step
             dt0 = max(dt -  time_counter, 0.0_kind_phys)
             do klev = lyr_surf, lyr_toa - lyr_step, lyr_step
                if (abs(velqr(klev)) > 1.0E-12_kind_phys) then
                   dt0 = min(dt0, 0.8_kind_phys*(z(col, klev+lyr_step) - z(col, klev)) / velqr(klev))
                end if
             end do

         end do

         ! compute the average preciptation rate over the physics time step period
         precl(col) = precl_acc / dt

         ! Diagnostic: relative humidity (relhum)
         do klev = lyr_surf, lyr_toa, lyr_step
            ! Saturation vapor mixing ratio (gm/gm)
            qvs = pc(klev) * exp(f2x*(pk(col, klev)*theta(col, klev) - 273._kind_phys) / (pk(col, klev)*theta(col, klev) &
                           - 36._kind_phys))
            relhum(col,klev) = qv(col,klev) / qvs * 100._kind_phys ! in percent
         end do

      end do ! column loop

   end subroutine kessler_run
