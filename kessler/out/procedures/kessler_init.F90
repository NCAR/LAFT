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
