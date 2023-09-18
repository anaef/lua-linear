LUA_INCDIR=/usr/include/lua5.3
LUA_BIN=/usr/bin/lua5.3
LIBDIR=/usr/local/lib/lua/5.3
CFLAGS=-Wall -Wextra -Wpointer-arith -Werror -fPIC -O3 -D_REENTRANT -D_GNU_SOURCE -DLUA_LINEAR_HAVE_AXPBY
LDFLAGS=-shared -fPIC

export LUA_CPATH=$(PWD)/?.so

default: all

all: linear.so

linear.so: linear.o
	gcc $(LDFLAGS) -o linear.so linear.o -lblas -llapacke

linear.o: src/linear.h src/linear.c
	gcc -c -o linear.o $(CFLAGS) -I$(LUA_INCDIR) src/linear.c

.PHONY: test
test:
	$(LUA_BIN) test/test.lua

install:
	cp linear.so $(LIBDIR)

clean:
	-rm -f linear.o linear.so
