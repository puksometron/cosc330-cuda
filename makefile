COMPILER = nvcc
CFLAGS = -I /usr/local/cuda-11.4/samples/common/inc 
EXES = program

all:
	${EXES}

program:	program.cu
	${COMPILER} ${CFLAGS} program.cu bmpfile.c -o program

clean:
	rm -f *.o *~ ${EXES} ${CFILES}

run:
	./program