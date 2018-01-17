TARGET	= jacobi
OBJS	= mainPois.o jacobi.o gauss_seidel.o Mallocation.o writepng.o

OPT	= -g -fast 
ISA	= 
PARA	= 

PNGWRITERPATH = pngwriter
ARCH	      = $(shell uname -p)
PNGWRTLPATH   = $(PNGWRITERPATH)/lib/$(ARCH)
PNGWRTIPATH   = $(PNGWRITERPATH)/include
PNGWRITERLIB  = $(PNGWRTLPATH)/libpngwriter.a

CCC	= CC
CXX	= CC
CXXFLAGS= -I $(PNGWRTIPATH)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

F90C  	= f90

LIBS	= -L $(PNGWRTLPATH) -lpngwriter -lpng 


all: $(PNGWRITERLIB) $(TARGET)

$(TARGET): $(OBJS) 
	$(CCC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

$(PNGWRITERLIB):
	@cd pngwriter/src && $(MAKE)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@cd pngwriter/src && $(MAKE) clean
	@rm -f $(PNGWRITERLIB)
	@rm -f $(TARGET)
	@rm -f jacobi.png

# dependencies
#
main.o  : mainPois.c jacobi.h gauss_seidel.h Mallocation.h writepng.h
writepng.o: writepng.h writepng.cc
jacobi.o: jacobi.h jacobi.c
gauss_seidel.o: gauss_seidel.h gauss_seidel.c
Mallocation.o: Mallocation.h Mallocation.c
