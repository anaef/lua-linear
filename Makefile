LUA_INCLUDE = /usr/include/lua5.2
LUA_BIN = /usr/bin/lua5.2
LIB_INSTALL = /usr/local/lib/lua/5.2
CFLAGS = -pedantic -Wall -Wextra -fPIC -O3 -D_REENTRANT -D_GNU_SOURCE
LDFLAGS = -shared -fPIC

export LUA_CPATH=$(PWD)/?.so

default: 
	@echo "Please run make <linux|macosx>."

linux:
	$(MAKE) all CFLAGS="$(CFLAGS) -fopenmp" LDFLAGS="$(LDFLAGS) -fopenmp"

macosx:
	$(MAKE) all LDFLAGS="$(LDFLAGS) -undefined dynamic_lookup"

all: linear.so test

test:
	$(LUA_BIN) test.lua

linear.so: linear.o
	gcc $(LDFLAGS) -o linear.so linear.o -lblas -llapacke

linear.o: linear.h linear.c
	gcc -c -o linear.o $(CFLAGS) -I$(LUA_INCLUDE) linear.c

install:
	cp linear.so $(LIB_INSTALL)

clean:
	-rm linear.o linear.so
