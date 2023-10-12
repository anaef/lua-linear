LUA_INCDIR=/usr/include/lua5.3
LUA=/usr/bin/lua5.3
LIBDIR=/usr/local/lib/lua/5.3
CFLAGS=-Wall -Wextra -Wpointer-arith -Werror -fPIC -O3 -D_REENTRANT -D_GNU_SOURCE
LDFLAGS=-shared -fPIC
USE_AXPBY=1
FEATURES=-DLINEAR_USE_AXPBY=$(USE_AXPBY) \

export LUA_CPATH=$(PWD)/?.so

default: all

all: linear.so

linear.so: linear_core.o linear_elementary.o linear_unary.o linear_binary.o linear_program.o
	gcc $(LDFLAGS) -o linear.so linear_core.o linear_elementary.o linear_unary.o \
			linear_binary.o linear_program.o -lm -lblas -llapacke

linear_core.o: src/linear_core.h src/linear_core.c
	gcc -c -o linear_core.o $(CFLAGS) $(FEATURES) -I$(LUA_INCDIR) src/linear_core.c

linear_elementary.o: src/linear_core.h src/linear_elementary.h src/linear_elementary.c
	gcc -c -o linear_elementary.o $(CFLAGS) $(FEATURES) -I$(LUA_INCDIR) src/linear_elementary.c

linear_unary.o: src/linear_core.h src/linear_unary.h src/linear_unary.c
	gcc -c -o linear_unary.o $(CFLAGS) $(FEATURES) -I$(LUA_INCDIR) src/linear_unary.c

linear_binary.o: src/linear_core.h src/linear_binary.h src/linear_binary.c
	gcc -c -o linear_binary.o $(CFLAGS) $(FEATURES) -I$(LUA_INCDIR) src/linear_binary.c

linear_program.o: src/linear_core.h src/linear_program.h src/linear_program.c
	gcc -c -o linear_program.o $(CFLAGS) $(FEATURES) -I$(LUA_INCDIR) src/linear_program.c

.PHONY: test
test:
	$(LUA) test/test.lua

install:
	cp linear.so $(LIBDIR)

clean:
	-rm -f linear_core.o linear_elementary.o linear_unary.o linear_binary.o linear_program.o linear.so
