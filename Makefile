# ==========================================
# Makefile for CVRP Solvers (RCPW & CCW)
# ==========================================

# Compilers
NVCC := nvcc
CXX  := g++

# Target Executables
TARGET_RCPW := rpcw_solver
TARGET_CCW  := CCW.out

# Compiler Flags
NVCC_FLAGS := -arch=sm_80 -O3
CXX_FLAGS  := -O3

# Default rule: build both solvers
all: $(TARGET_RCPW) $(TARGET_CCW)

RPCW: $(TARGET_RCPW)
CCW: $(TARGET_CCW)

# Build the GPU-accelerated RCPW solver
$(TARGET_RCPW): RCPW.cu KDTree.cpp
	$(NVCC) $(NVCC_FLAGS) RCPW.cu KDTree.cpp -o $@

# Build the CPU baseline CCW solver
$(TARGET_CCW): CCW.cpp
	$(CXX) $(CXX_FLAGS) CCW.cpp -o $@

# Clean up build artifacts
clean:
	rm -f $(TARGET_RCPW) $(TARGET_CCW)

.PHONY: all clean