LUA_INCLUDE=/usr/include/lua5.3
LUA_BIN=/usr/bin/lua5.3
LIB_INSTALL=/usr/local/lib/lua/5.3
CFLAGS=-Wall -Wextra -Wpointer-arith -Werror -fPIC -fopenmp -O3 -D_REENTRANT -D_GNU_SOURCE
LDFLAGS=-shared -fPIC -fopenmp

export LUA_CPATH=$(PWD)/?.so

default: all test

linear.o: linear.h linear.c
	gcc -c -o linear.o $(CFLAGS) -I$(LUA_INCLUDE) linear.c

linear.so: linear.o
	gcc $(LDFLAGS) -o linear.so linear.o -lblas -llapacke

test:
	$(LUA_BIN) test.lua

all: linear.so

install:
	cp linear.so $(LIB_INSTALL)

clean:
	-rm linear.o linear.so
