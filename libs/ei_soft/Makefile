CC=gcc
FILEC=maxfilecompile
CFLAGS=-O3 -std=c99 
CFLAGS2=-parallel -fast -O3

python: 
	swig -python ei_multi.i
	gcc -fPIC -std=c99 -c ei_multi.c -lm 
	gcc -fPIC -std=c99 -c ei_multi_wrap.c -lpython2.7 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include
	ld -shared ei_multi_wrap.o ei_multi.o -o _ei_soft.so

all: ei_bench

$ei_bench.o: ei_bench.c
		${CC} $< ${CFLAGS} -lm -o $@
		
$ei_bench: ei_bench.o
	${CC} -o
        
clean:
		rm -f ei_bench
		
realclean: clean
		rm -fr ei_bench
		rm -f ei_bench.o
