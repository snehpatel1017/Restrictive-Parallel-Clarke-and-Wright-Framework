# ==========================================
# Makefile for RPCW CVRP Solver
# ==========================================

# Compilers
NVCC := nvcc
CXX  := g++

# Target Executable Name
TARGET := rpcw_solver

# GPU Architecture (sm_80 is for NVIDIA A100. Change to sm_75 for Turing or sm_86 for Ampere/RTX 30-series)
ARCH := sm_80

# Compiler Flags
NVCC_FLAGS := -O3 -arch=$(ARCH) -std=c++17
CXX_FLAGS  := -O3 -Wall -Wextra -std=c++17

# Source Files
CUDA_SRC := RCPW.cu
CPP_SRC  := CCW.cpp KDTree.cpp

# Object Files
CUDA_OBJ := $(CUDA_SRC:.cu=.o)
CPP_OBJ  := $(CPP_SRC:.cpp=.o)
OBJS     := $(CUDA_OBJ) $(CPP_OBJ)

# Default rule
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compiling CUDA source
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compiling C++ source
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean