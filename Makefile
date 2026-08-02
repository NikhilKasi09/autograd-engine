CC ?= gcc

# -mavx2 and -mfma are critical for Seyaan's kernel
# -pthread is critical for the multithreaded wrapper
CFLAGS ?= -Wall -Wextra -Werror -pedantic -O3 -mavx2 -mfma -pthread -Iinclude -MMD -MP

LDFLAGS = -lm

TARGET      = gemm_benchmark
TEST_TARGET = gemm_tests

# The kernel library on its own, so the test binary can link it without pulling
# in a second main(). main.c lives at the repo root rather than src/, so the
# wildcard leaves it out already.
LIB_SRCS  = $(wildcard src/*.c)
LIB_OBJS  = $(patsubst %.c,%.o,$(LIB_SRCS))

TEST_SRCS = $(wildcard tests/*.c)
TEST_OBJS = $(patsubst %.c,%.o,$(TEST_SRCS))

OBJS = main.o $(LIB_OBJS)
DEPS = $(OBJS:.o=.d) $(TEST_OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST_TARGET): $(LIB_OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Lets the tsan target wrap the test binary (see below). Empty otherwise.
RUNNER =

test: $(TEST_TARGET)
	$(RUNNER) ./$(TEST_TARGET)

# Sanitiser builds. LDFLAGS has to be overridden by hand as well as CFLAGS,
# since the sanitisers are needed at link time too. Both targets clean first,
# or stale uninstrumented objects get linked in silently.
# -O1 keeps naive down to a couple of seconds under instrumentation, and
# -fno-sanitize-recover makes UBSan fail rather than print and carry on.
SAN_CFLAGS = -Wall -Wextra -Werror -pedantic -g -O1 -fno-omit-frame-pointer \
             -mavx2 -mfma -pthread -Iinclude -MMD -MP

asan:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(SAN_CFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=all" \
	        LDFLAGS="-lm -fsanitize=address,undefined" test

# TSan cannot combine with ASan, so it gets its own target. This is the one
# that catches two workers writing to overlapping rows.
#
# setarch -R turns off ASLR for the run. Ubuntu 24.04 defaults to
# vm.mmap_rnd_bits=32 but TSan only handles up to 28, so without it the binary
# segfaults or dies with "unexpected memory mapping", intermittently depending
# on where the loader mapped things. sysctl vm.mmap_rnd_bits=28 fixes it
# system-wide instead.
tsan:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(SAN_CFLAGS) -fsanitize=thread -fno-sanitize-recover=all" \
	        LDFLAGS="-lm -fsanitize=thread" \
	        RUNNER="setarch $(shell uname -m) -R" test

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(DEPS) $(TARGET) $(TEST_TARGET)

.PHONY: all clean test asan tsan

-include $(DEPS)
