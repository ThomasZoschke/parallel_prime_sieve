# parallel prime number sieve Makefile
.PHONY: all clean profiled


# Do not use make's built-in rules and variables
# (this increases performance and avoids hard-to-debug behaviour);
MAKEFLAGS += -rR
CC := g++-5
OPTARGS := -fopenmp -lgomp
OPTARGS += -m64 -O4 
OPTARGS += -W -Wall
OPTARGS += -std=gnu++11 -ffast-math
OPTARGS += -fexpensive-optimizations -fno-exceptions -fno-rtti 
OPTARGS += -march=core2 -mtune=core2 -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1
OPTARGS += -ffunction-sections -Wl,-O1,-init=_start,--gc-sections

TARGETS := 
TARGETS += prime-numbers
TARGETS += prime-numbers.debug
TARGETS += prime-numbers.genprof
TARGETS += prime-numbers.profiled
TARGETS += prime-numbers.gcda
TARGETS += prime-number-sieve

all:
	@make -j 2 -k prime-numbers.debug prime-numbers

prime-numbers.debug: prime-numbers.cpp Makefile
	$(CC) -m64 -g -DDEBUG=1 -fopenmp -lgomp -std=gnu++11 -W -Wall -o $@ $<

prime-numbers: prime-numbers.cpp Makefile
	$(CC) $(OPTARGS) $(PGUARG) -o $@ $<

profiled: prime-numbers.gcda
	rm -f prime-numbers
	make prime-numbers PGUARG="--profile-use"

prime-numbers.gcda: prime-numbers.cpp Makefile
	rm -f $@
	rm -f prime-numbers
	make prime-numbers PGUARG="--profile-generate"
	echo 99999999 | ./prime-numbers
	test -f $@

clean:
	for i in $(TARGETS); \
	do if test -f "$$i"; \
	then rm -i "$$i"; \
	fi; \
	done
