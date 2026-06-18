CC ?= gcc

CFLAGS ?= -Wall -Wextra -Werror -pedantic -O3 -mavx2 -mfma -pthread -Iinclude -MMD -MP

LDFLAGS = -lm

TARGET = gemm_benchmark

SRCS = main.c $(wildcard src/*.c)
OBJS = $(SRCS:.c=.o)
DEPS = $(SRCS:.c=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean

-include $(DEPS)