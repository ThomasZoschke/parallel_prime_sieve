# parallel prime nuber sieve Makefile
.PHONY: all clean


# Do not use make's built-in rules and variables
# (this increases performance and avoids hard-to-debug behaviour);
#MAKEFLAGS += -rR


all:
	@make -j 2 -k prime-numbers.debug prime-numbers

prime-numbers.debug: prime-numbers.cpp Makefile
	g++-5 -m64 -g -DDEBUG=1 -W -Wall -fopenmp -lgomp -std=gnu++11 -o $@ $<

prime-numbers: prime-numbers.cpp Makefile
	g++-5 -m64 -O4 -W -Wall --profile-use \
		 -fopenmp -lgomp -std=gnu++11 -fexpensive-optimizations -fno-exceptions -fno-rtti \
		 -march=core2 -mtune=core2 -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -ffast-math \
		 -ffunction-sections -Wl,-O1,-init=_start,--gc-sections \
		 -o $@ $<

clean:
	for i in prime-numbers prime-numbers.debug prime-number-sieve; \
	do if test -f "$$i"; \
	then rm -i "$$i"; \
	fi; \
	done
