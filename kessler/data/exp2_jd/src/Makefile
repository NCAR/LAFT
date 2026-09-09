# ============================================================
# Minimal Fortran Makefile (explicit ordering, no magic)
# ============================================================

#COMPILER ?= gnu
COMPILER ?= nvhpc

# ------------------------------------------------------------
# Compiler selection
# ------------------------------------------------------------

ifeq ($(COMPILER),gnu)
  FC = gfortran
  FFLAGS = -O3 -Wall -cpp -fopenmp -ffree-line-length-none
endif

ifeq ($(COMPILER),nvhpc)
  FC = nvfortran
  ifeq ($(ARCH),GPU)
    FFLAGS = -O3 -Mpreprocess -acc -mp=gpu -gpu=mem:separate,math_uniform -DUSE_GPU
  else
    FFLAGS = -O3 -Mpreprocess
  endif
  FFLAGS += -Mnofma
endif

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

TARGET = JD_kessler_driver

# ------------------------------------------------------------
# Build (EXPLICIT ORDER — THIS IS THE KEY)
# ------------------------------------------------------------

all: $(TARGET)

compare:
	./$(TARGET) >& test.$(COMPILER).log
	diff test.$(COMPILER).log golden-nofma.$(COMPILER).log

$(TARGET): ccpp_kinds.o kessler.o kessler_update.o JD_kessler_driver.o
	$(FC) $(FFLAGS) -o $(TARGET) \
	    ccpp_kinds.o kessler.o kessler_update.o JD_kessler_driver.o

# Compile in strict dependency order

ccpp_kinds.o: ccpp_kinds.F90
	$(FC) $(FFLAGS) -c ccpp_kinds.F90

kessler.o: kessler.F90 ccpp_kinds.o
	$(FC) $(FFLAGS) -c kessler.F90

kessler_update.o: kessler_update.F90 ccpp_kinds.o
	$(FC) $(FFLAGS) -c kessler_update.F90

JD_kessler_driver.o: JD_kessler_driver.F90 kessler.o kessler_update.o
	$(FC) $(FFLAGS) -c JD_kessler_driver.F90

# ------------------------------------------------------------

clean:
	rm -f *.o *.mod $(TARGET)

.PHONY: all clean compare
