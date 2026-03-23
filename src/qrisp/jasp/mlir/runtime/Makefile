# Makefile for the JASP C runtime
#
# Set QDMI_ROOT to the path of your QDMI checkout before building.
# Example:
#   make QDMI_ROOT=/home/user/git/QDMI

QDMI_ROOT ?= /home/bca/git/QDMI

CC       := clang
CFLAGS   := -std=c11 -Wall -Wextra
INCLUDES := -I. \
            -I$(QDMI_ROOT)/include \
            -I$(QDMI_ROOT)/examples/driver
LIBS     := $(QDMI_ROOT)/build/examples/driver/libqdmi_example_driver.a \
            -ldl -lstdc++

.PHONY: all clean

all: runtime.o

# Compile runtime as an object file (link into your EmitC output later)
runtime.o: runtime.c runtime.h runtime_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c runtime.c -o runtime.o

# Example: link EmitC-generated output.c with the runtime
# Usage: make program OUTPUT=output.c
program: runtime.o
	$(CC) $(CFLAGS) $(INCLUDES) $(OUTPUT) runtime.o $(LIBS) -o program

clean:
	rm -f runtime.o program
