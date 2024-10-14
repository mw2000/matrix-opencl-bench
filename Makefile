# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm -framework OpenCL

# Directories
LIBDIR = lib
OBJDIR = obj
BINDIR = bin

# Source and object files
SRC = main.c $(LIBDIR)/matrix_cpu.c $(LIBDIR)/matrix_opencl.c
OBJ = $(OBJDIR)/main.o $(OBJDIR)/matrix_cpu.o $(OBJDIR)/matrix_opencl.o

# Target executable
TARGET = $(BINDIR)/matrix_opencl_bench

# Create object and binary directories if they don't exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Build the target
all: $(TARGET)

# Rule to build the target from object files
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

# Rule to compile main.c into obj/main.o
$(OBJDIR)/main.o: main.c
	$(CC) $(CFLAGS) -I$(LIBDIR) -c $< -o $@

# Rule to compile lib/matrix_cpu.c into obj/matrix_cpu.o
$(OBJDIR)/matrix_cpu.o: $(LIBDIR)/matrix_cpu.c
	$(CC) $(CFLAGS) -I$(LIBDIR) -c $< -o $@

# Rule to compile lib/matrix_opencl.c into obj/matrix_opencl.o
$(OBJDIR)/matrix_opencl.o: $(LIBDIR)/matrix_opencl.c
	$(CC) $(CFLAGS) -I$(LIBDIR) -c $< -o $@

# Rule to run the compiled program
run: $(TARGET)
	@echo "Running the program..."
	@./$(TARGET)

# Clean up object files and the binary
clean:
	rm -rf $(OBJDIR) $(BINDIR)
