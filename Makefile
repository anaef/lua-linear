LUA_INCLUDE=/usr/include/lua5.3
LUA_BIN=/usr/bin/lua5.3
LIB_INSTALL=/usr/local/lib/lua/5.3
CFLAGS=-Wall -Wextra -Wpointer-arith -Werror -fPIC -O3 -D_REENTRANT -D_GNU_SOURCE
LDFLAGS=-shared -fPIC

export LUA_CPATH=$(PWD)/?.so

default: all test

linear.o: src/linear.h src/linear.c
	gcc -c -o linear.o $(CFLAGS) -I$(LUA_INCLUDE) src/linear.c

linear.so: linear.o
	gcc $(LDFLAGS) -o linear.so linear.o -lblas -llapacke

all: linear.so

.PHONY: test
test:
	$(LUA_BIN) test/test.lua

install:
	cp linear.so $(LIB_INSTALL)

clean:
	-rm -f linear.o linear.so
