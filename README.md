# RPCW: Restrictive Parallel Clarke-Wright Solver

[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The **Restrictive Parallel Clarke-Wright (RPCW)** solver is a massively parallel, CUDA-accelerated heuristic designed to solve extreme-scale Capacitated Vehicle Routing Problem (CVRP) instances. By leveraging iterative graph-sparsification and bipartite-like parallel matching, this solver eliminates the traditional sorting bottlenecks of the sequential Clarke-Wright algorithm, capable of processing up to 1,000,000 customers in near real-time.

---

## 📁 Repository Structure

```text
.
├── Instances/           # CVRP Benchmark Datasets
│   ├── Belgium/         # Large-scale real-world instances
│   ├── Golden/          # Classical benchmark instances
│   ├── I/               # Ultra-large-scale Set I (up to 1M nodes)
│   ├── Others/          # Mixed benchmark families (CMT, Fisher, etc.)
│   └── X/               # Uchoa et al. benchmark set
├── CCW.cpp              # CPU baseline implementation
├── KDTree.cpp           # K-Dimensional Tree source for sparsification
├── KDTree.hpp           # K-Dimensional Tree header
├── RCPW.cu              # Core GPU/CUDA implementation
├── toy.vrp              # Sample instance for immediate testing
└── Makefile             # Build automation script
```

---

## 🛠 System Requirements

To compile and run this code, ensure your system meets the following requirements:

### Hardware
- An NVIDIA GPU with Compute Capability **7.0 or higher** (Tested on NVIDIA A100).
- Sufficient system RAM to load large `.vrp` instances (**16 GB+ recommended** for 1M node datasets).

### Software
- **OS:** Linux (Ubuntu 20.04 / 22.04 recommended)
- **NVIDIA Driver:** version 550.x or compatible
- **CUDA Toolkit:** version 12.0+ (Tested on 12.4)
- **C++ Compiler:** `g++` (GCC) with C++17 support
- **Build Tool:** `make`

---

## 🚀 Installation & Build Instructions

### 1. Clone the repository

Download the source code to your local machine:

```bash
git clone https://github.com/snehpatel1017/Restrictive-Parallel-Clarke-and-Wright-Framework.git
cd Restrictive-Parallel-Clarke-and-Wright-Framework
```

### 2. Compile the solver

The project uses a `Makefile` to streamline the build process. By default, the Makefile targets the `sm_80` architecture (A100). If you are using a different GPU (e.g., RTX 3090 / `sm_86`), update the `ARCH` variable in the Makefile before compiling.

Run the following command to compile:

```bash
make
```

This will compile the C++ and CUDA source files and link them into an executable named `rpcw_solver`.

### 3. Clean the build (Optional)

To remove the compiled object files and the executable, run:

```bash
make clean
```

---

## ⚡ Running the Solver

Once compiled, you can run the solver by passing a `.vrp` instance file as an argument.

**Run a quick test** using the included toy instance:

```bash
./rpcw_solver toy.vrp
```

**Run a large-scale benchmark instance:**

```bash
./rpcw_solver Instances/I/Lazio.vrp
```

> **Note:** Depending on how your argument parsing is configured in `main()`, you may need to append additional flags for hyperparameters like $K$.

---

## 🧠 Core Components

| Component | File(s) | Description |
|---|---|---|
| **KD-Tree Sparsification** | `KDTree.cpp` / `KDTree.hpp` | Rapidly constructs the initial $K$-nearest neighbor graph to prune non-viable long-distance edges. |
| **RPCW GPU Pipeline** | `RCPW.cu` | Contains the highly parallel kernels (`k1`, `k2`, `k3`) for proposal generation, mutual selection, and memory-safe route merging. |
| **Sequential Baseline** | `CCW.cpp` | Standard CPU implementation provided for performance and quality comparison. |