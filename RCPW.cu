#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cfloat>  // For DBL_MAX
#include <iomanip> // For std::setprecision
#include <chrono>  // For timing
#include <omp.h>
#include <stdio.h>
#include <utility>
#include <set>
#include "KDTree.hpp"
// CUDA specific headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;

const node_t DEPOT = 0;

struct Point
{
    double x, y, demand;
};

class VRP
{
public:
    size_t size;
    demand_t capacity;
    std::vector<Point> node;
    std::vector<weight_t> dist_to_depot;
    static bool isRound;
    static int K;

    VRP() : size(0), capacity(0) {}

    void read(const std::string &filename);
    weight_t get_dist(node_t i, node_t j) const;

    size_t getSize() const
    {
        return size;
    }
    demand_t getCapacity() const
    {
        return capacity;
    }
};

bool VRP::isRound = false;
int VRP::K = 20;

void VRP::read(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (getline(in, line) && line.find("DIMENSION") == std::string::npos)
        ;
    if (line.find(":") != std::string::npos)
        size = stoul(line.substr(line.find(":") + 1));
    while (getline(in, line) && line.find("CAPACITY") == std::string::npos)
        ;
    if (line.find(":") != std::string::npos)
        capacity = stoul(line.substr(line.find(":") + 1));
    while (getline(in, line) && line.find("NODE_COORD_SECTION") == std::string::npos)
        ;
    node.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        int id;
        in >> id >> node[i].x >> node[i].y;
    }
    while (getline(in, line) && line.find("DEMAND_SECTION") == std::string::npos)
        ;
    for (size_t i = 0; i < size; ++i)
    {
        int id;
        in >> id >> node[i].demand;
    }
    in.close();
    dist_to_depot.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        dist_to_depot[i] = get_dist(DEPOT, i);
    }
}

weight_t VRP::get_dist(node_t i, node_t j) const
{
    double dx = node[i].x - node[j].x;
    double dy = node[i].y - node[j].y;
    double dist = sqrt(dx * dx + dy * dy);
    if (isRound)
        return std::round(dist);
    return dist;
}

weight_t calCost(const VRP &vrp, const std::vector<std::vector<node_t>> &routes)
{
    weight_t total_cost = 0.0;
    for (const auto &route : routes)
    {
        if (route.empty())
            continue;

        node_t last_node = DEPOT;
        for (node_t current_node : route)
        {
            total_cost += vrp.get_dist(last_node, current_node);
            last_node = current_node;
        }
        // Add cost to return to the depot
        total_cost += vrp.get_dist(last_node, DEPOT);
    }
    return total_cost;
}

bool verify_sol(const VRP &vrp, std::vector<std::vector<node_t>> final_routes, unsigned capacity)
{
    /* verifies if the solution is valid or not */
    /**
     * 1. All vertices appear in the solution exactly once.
     * 2. For every route, the capacity constraint is respected.
     **/

    unsigned *hist = (unsigned *)malloc(sizeof(unsigned) * vrp.getSize());
    memset(hist, 0, sizeof(unsigned) * vrp.getSize());

    for (unsigned i = 0; i < final_routes.size(); ++i)
    {
        unsigned route_sum_of_demands = 0;
        for (unsigned j = 0; j < final_routes[i].size(); ++j)
        {
            //~ route_sum_of_demands += points.demands[final_routes[i][j]];
            route_sum_of_demands += vrp.node[final_routes[i][j]].demand;
            hist[final_routes[i][j]] += 1;
        }
        if (route_sum_of_demands > capacity)
        {
            return false;
        }
    }

    for (unsigned i = 1; i < vrp.getSize(); ++i)
    {
        if (hist[i] > 1)
        {
            return false;
        }
        if (hist[i] == 0)
        {
            return false;
        }
    }
    return true;
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
                  << " \"" << cudaGetErrorString(result) << "\" for " << func << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

__device__ double device_euclidean_dist(const double aX, const double aY, const double bX, const double bY)
{
    return sqrt((aX - bX) * (aX - bX) + (aY - bY) * (aY - bY));
}

__device__ volatile unsigned int global_counter = 0;

__device__ double atomicMax(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        // Compare double values and use CAS to update if the new value is larger
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));

        // Note: uses integer comparison to detect if the value was changed by another thread
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double atomicMin(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        // Compare double values and use CAS to update if the new value is smaller
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));

        // Uses integer comparison to detect if the memory was modified by another thread
    } while (assumed != old);

    return __longlong_as_double(old);
}
__global__ void k1(
    const node_t *edges_X,
    const node_t *edges_Y,
    const weight_t *edges_W,
    weight_t *best_saving,
    const weight_t *route_demands,
    const node_t *route_head,
    const node_t *route_tail,
    demand_t capacity,
    unsigned int last_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    unsigned int limit = last_index;
    for (int i = tid; i < limit; i += total_threads)
    {
        node_t a = edges_X[i];
        node_t b = edges_Y[i];
        if (edges_W[i] <= 0)
            continue;

        node_t cr = route_head[b];
        if (route_demands[cr] + route_demands[route_head[a]] > capacity)
            continue;

        atomicMax(&best_saving[a], edges_W[i]);
        atomicMax(&best_saving[cr], edges_W[i]);
    }
}

__global__ void k2(
    const node_t *edges_X,
    const node_t *edges_Y,
    const weight_t *edges_W,
    weight_t *best_saving,
    demand_t *best_demand,
    const weight_t *route_demands,
    const node_t *route_head,
    const node_t *route_tail,
    demand_t capacity,
    unsigned int last_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    unsigned int limit = last_index;
    for (int i = tid; i < limit; i += total_threads)
    {
        node_t a = edges_X[i];
        node_t b = edges_Y[i];

        if (edges_W[i] <= 0)
            continue;
        node_t cr = route_head[b];
        demand_t tot = route_demands[cr] + route_demands[route_head[a]];
        if (tot > capacity)
            continue;
        if (best_saving[a] == edges_W[i])
            atomicMin(&best_demand[a], tot);
        if (best_saving[cr] == edges_W[i])
            atomicMin(&best_demand[cr], tot);
    }
}

__global__ void k3(const node_t *edges_X,
                   const node_t *edges_Y,
                   const weight_t *edges_W,
                   weight_t *best_saving,
                   demand_t *best_demand,
                   node_t *crush,
                   const weight_t *route_demands,
                   const node_t *route_head,
                   const node_t *route_tail,
                   demand_t capacity,
                   unsigned int last_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    unsigned int limit = last_index;
    for (int i = tid; i < limit; i += total_threads)
    {
        node_t a = edges_X[i];
        node_t b = edges_Y[i];

        if (edges_W[i] <= 0)
            continue;
        node_t cr = route_head[b];
        demand_t tot = route_demands[cr] + route_demands[route_head[a]];
        if (tot > capacity || best_saving[a] != best_saving[cr] || best_saving[a] != edges_W[i] || best_demand[a] != tot || best_demand[cr] != tot)
            continue;

        atomicMin(&crush[a], cr);
        atomicMin(&crush[cr], a);
    }
}

__global__ void get_pairs(
    const double *X,
    const double *Y,
    node_t *route_head,
    node_t *route_tail,
    node_t *crush,
    const weight_t *dist_to_depot,
    node_t *store_i,
    node_t *store_j,
    unsigned int last_index,
    unsigned int *holding_global_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i <= last_index; i += total_threads)
    {
        if (crush[i] == last_index + 2)
            continue;

        int j = crush[i];
        if (crush[j] != i)
            continue;

        node_t route_id_i = i;
        node_t route_id_j = j;
        node_t head_i = route_head[route_id_i];
        node_t tail_i = route_tail[route_id_i];
        node_t head_j = route_head[route_id_j];
        node_t tail_j = route_tail[route_id_j];
        double saving_1 = dist_to_depot[tail_i] + dist_to_depot[head_j] - device_euclidean_dist(X[tail_i], Y[tail_i], X[head_j], Y[head_j]);
        double saving_2 = dist_to_depot[tail_j] + dist_to_depot[head_i] - device_euclidean_dist(X[tail_j], Y[tail_j], X[head_i], Y[head_i]);
        if (saving_1 < saving_2)
        {
            continue;
        }
        if (saving_1 == saving_2)
        {
            if (i > j)
                continue;
        }
        int old_pos = atomicAdd((unsigned int *)holding_global_counter, (unsigned int)1);
        store_i[old_pos] = i;
        store_j[old_pos] = j;
    }
}

__global__ void merging(
    node_t *store_i,
    node_t *store_j,
    weight_t *route_demands,
    node_t *route_head,
    node_t *route_tail,
    node_t *next_customer, unsigned int *holding_global_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int curr = tid; curr < *holding_global_counter; curr += total_threads)
    {
        node_t i = store_i[curr];
        node_t j = store_j[curr];
        node_t route_id_i = i;
        node_t route_id_j = j;
        node_t tail_i = route_tail[route_id_i];
        node_t head_j = route_head[route_id_j];
        node_t tail_j = route_tail[route_id_j];
        next_customer[tail_i] = head_j;
        route_tail[route_id_i] = tail_j;
        route_head[tail_j] = route_id_i;
        route_demands[route_id_i] += route_demands[route_id_j];
        route_demands[route_id_j] = 0;
        route_head[route_id_j] = route_id_i;
    }
}

__global__ void edge_cleanup(
    node_t *edges_X,
    node_t *edges_Y,
    weight_t *edges_W,
    node_t *temp_edges_X,
    node_t *temp_edges_Y,
    weight_t *temp_edges_W,
    const node_t *route_head,
    const node_t *route_tail,
    unsigned int last_index,
    unsigned int *slow_pointer)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int i = tid; i < last_index; i += total_threads)
    {
        node_t a = edges_X[i];
        node_t b = edges_Y[i];
        if (route_head[a] == route_head[b])
            continue;
        if (a != route_head[a] && route_tail[route_head[a]] != a)
            continue;
        if (b != route_head[b] && route_tail[route_head[b]] != b)
            continue;
        if ((route_head[b] == b && route_tail[route_head[a]] == a) || (route_head[a] == a && route_tail[route_head[b]] == b))
        {
            if ((route_head[b] == b && route_tail[route_head[a]] == a))
            {
                node_t temp = a;

                a = b;
                b = temp;
            }

            unsigned int pos = atomicAdd((unsigned int *)slow_pointer, 1);
            temp_edges_X[pos] = a;
            temp_edges_Y[pos] = b;
            temp_edges_W[pos] = edges_W[i];
        }
    }
}
__global__ void reset(
    node_t *crush,
    weight_t *best_saving,
    demand_t *best_demand,
    unsigned int last_index,
    const demand_t CAPACITY,
    unsigned int *holding_global_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    if (tid == 0)
    {
        *holding_global_counter = 0;

        global_counter = 0;
    }
    for (int i = tid; i <= last_index; i += total_threads)
    {
        if (i == 0)
            continue;
        crush[i] = last_index + 2;
        best_saving[i] = 0;
        best_demand[i] = CAPACITY + 1;
    }
}

std::vector<std::vector<std::pair<node_t, node_t>>> mergings;
std::vector<std::vector<node_t>> RCPW(const VRP &vrp, std::vector<node_t> &edges_X, std::vector<node_t> &edges_Y, std::vector<weight_t> &edges_W)
{
    const int NUM_CUSTOMERS = vrp.getSize() - 1; // Exclude depot
    const demand_t CAPACITY = vrp.getCapacity();
    std::vector<double> h_X(NUM_CUSTOMERS + 1);
    std::vector<double> h_Y(NUM_CUSTOMERS + 1);
    std::vector<weight_t> best_saving(NUM_CUSTOMERS + 1, 0);
    std::vector<demand_t> best_demand(NUM_CUSTOMERS + 1, CAPACITY + 1);
    std::vector<demand_t> h_route_demands(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_head(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_route_tail(NUM_CUSTOMERS + 1);
    std::vector<node_t> h_next_customer(vrp.size, DEPOT);
    std::vector<node_t> h_crush(vrp.size, NUM_CUSTOMERS + 2);
    std::vector<node_t> h_store_i((NUM_CUSTOMERS) / 2 + 1, -1);
    std::vector<node_t> h_store_j((NUM_CUSTOMERS) / 2 + 1, -1);
    std::vector<double> kernel_times(7, 0.0);
    unsigned int h_slow_pointer = 0;

    for (int i = 1; i <= NUM_CUSTOMERS; ++i)
    {
        // Initially, each customer is in their own route
        h_X[i] = vrp.node[i].x;
        h_Y[i] = vrp.node[i].y;
        h_route_demands[i] = vrp.node[i].demand;
        h_route_head[i] = i;
        h_route_tail[i] = i;
    }

    // --- 2. DEVICE: Allocate GPU memory ---
    double *d_X;
    double *d_Y;
    node_t *d_edges_X, *d_edges_Y;
    node_t *d_temp_edges_X, *d_temp_edges_Y;
    weight_t *d_edges_W, *d_temp_edges_W;
    weight_t *d_best_saving;
    demand_t *d_best_demand;
    demand_t *d_route_demands;
    node_t *d_route_head;
    node_t *d_route_tail;
    weight_t *d_dist_to_depot;
    node_t *d_next_customer;
    node_t *d_crush;
    node_t *d_store_i;
    node_t *d_store_j;
    unsigned int *d_slow_pointer;
    unsigned int *d_holding_global_counter;

    dim3 threadsPerBlock(1024);
    dim3 numBlocks((int)(NUM_CUSTOMERS + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&d_X, (NUM_CUSTOMERS + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_Y, (NUM_CUSTOMERS + 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_edges_X, NUM_CUSTOMERS * vrp.K * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_temp_edges_X, NUM_CUSTOMERS * vrp.K * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_edges_Y, NUM_CUSTOMERS * vrp.K * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_temp_edges_Y, NUM_CUSTOMERS * vrp.K * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_edges_W, NUM_CUSTOMERS * vrp.K * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_temp_edges_W, NUM_CUSTOMERS * vrp.K * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_best_saving, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_best_demand, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_demands, (NUM_CUSTOMERS + 1) * sizeof(demand_t)));
    checkCudaErrors(cudaMalloc(&d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_route_tail, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_dist_to_depot, (NUM_CUSTOMERS + 1) * sizeof(weight_t)));
    checkCudaErrors(cudaMalloc(&d_next_customer, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_crush, (NUM_CUSTOMERS + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_store_i, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_store_j, ((NUM_CUSTOMERS) / 2 + 1) * sizeof(node_t)));
    checkCudaErrors(cudaMalloc(&d_slow_pointer, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_holding_global_counter, sizeof(unsigned int)));

    // --- 3. Copy data from host to device ---
    checkCudaErrors(cudaMemcpy(d_X, h_X.data(), (NUM_CUSTOMERS + 1) * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Y, h_Y.data(), (NUM_CUSTOMERS + 1) * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_edges_X, edges_X.data(), edges_X.size() * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_edges_Y, edges_Y.data(), edges_X.size() * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_edges_W, edges_W.data(), edges_X.size() * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_best_saving, best_saving.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_best_demand, best_demand.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_demands, h_route_demands.data(), (NUM_CUSTOMERS + 1) * sizeof(demand_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_head, h_route_head.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_route_tail, h_route_tail.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dist_to_depot, vrp.dist_to_depot.data(), (NUM_CUSTOMERS + 1) * sizeof(weight_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_next_customer, h_next_customer.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_crush, h_crush.data(), (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slow_pointer, &h_slow_pointer, sizeof(unsigned int), cudaMemcpyHostToDevice));

    unsigned int last_index = edges_X.size();

    while (true)
    {

        k1<<<numBlocks, threadsPerBlock>>>(
            d_edges_X,
            d_edges_Y,
            d_edges_W,
            d_best_saving,
            d_route_demands,
            d_route_head,
            d_route_tail,
            CAPACITY,
            last_index);

        k2<<<numBlocks, threadsPerBlock>>>(
            d_edges_X,
            d_edges_Y,
            d_edges_W,
            d_best_saving,
            d_best_demand,
            d_route_demands,
            d_route_head,
            d_route_tail,
            CAPACITY,
            last_index);

        k3<<<numBlocks, threadsPerBlock>>>(
            d_edges_X,
            d_edges_Y,
            d_edges_W,
            d_best_saving,
            d_best_demand,
            d_crush,
            d_route_demands,
            d_route_head,
            d_route_tail,
            CAPACITY,
            last_index);

        get_pairs<<<numBlocks, threadsPerBlock>>>(
            d_X,
            d_Y,
            d_route_head,
            d_route_tail,
            d_crush,
            d_dist_to_depot,
            d_store_i,
            d_store_j,
            NUM_CUSTOMERS,
            d_holding_global_counter);

        merging<<<numBlocks, threadsPerBlock>>>(
            d_store_i,
            d_store_j,
            d_route_demands,
            d_route_head,
            d_route_tail,
            d_next_customer, d_holding_global_counter);

        edge_cleanup<<<numBlocks, threadsPerBlock>>>(
            d_edges_X,
            d_edges_Y,
            d_edges_W,
            d_temp_edges_X,
            d_temp_edges_Y,
            d_temp_edges_W,
            d_route_head,
            d_route_tail,
            last_index,
            d_slow_pointer);

        reset<<<numBlocks, threadsPerBlock>>>(
            d_crush,
            d_best_saving,
            d_best_demand,
            NUM_CUSTOMERS,
            CAPACITY,
            d_holding_global_counter);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(&h_slow_pointer, d_slow_pointer, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::swap(d_edges_X, d_temp_edges_X);
        std::swap(d_edges_Y, d_temp_edges_Y);
        std::swap(d_edges_W, d_temp_edges_W);

        if (last_index == h_slow_pointer)
        {

            break; // Exit the while loop
        }

        last_index = h_slow_pointer;
        h_slow_pointer = 0;
        checkCudaErrors(cudaMemcpy(d_slow_pointer, &h_slow_pointer, sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMemcpy(h_route_head.data(), d_route_head, (NUM_CUSTOMERS + 1) * sizeof(node_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_next_customer.data(), d_next_customer, vrp.size * sizeof(node_t), cudaMemcpyDeviceToHost));

    // --- 5. Finalize Routes ---
    std::vector<std::vector<node_t>> final_routes;

    for (node_t i = 1; i <= NUM_CUSTOMERS; ++i)
    {

        node_t current_node = h_route_head[i];
        if (current_node != i)
            continue;
        std::vector<node_t> current_route;
        while (current_node != DEPOT)
        {

            current_route.push_back(current_node);
            current_node = h_next_customer[current_node];
        }
        if (!current_route.empty())
        {
            final_routes.push_back(current_route);
        }
    }

    checkCudaErrors(cudaDeviceReset());

    return final_routes;
}

void tsp_approx(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, node_t ncities)
{
    node_t i, j;
    node_t ClosePt = 0;
    weight_t CloseDist;

    //~ node_t endtour=0;

    for (i = 1; i < ncities; i++)
        tour[i] = cities[i - 1];

    tour[0] = cities[ncities - 1];

    for (i = 1; i < ncities; i++)
    {
        weight_t ThisX = vrp.node[tour[i - 1]].x;
        weight_t ThisY = vrp.node[tour[i - 1]].y;
        CloseDist = DBL_MAX;
        for (j = ncities - 1;; j--)
        {
            weight_t ThisDist = (vrp.node[tour[j]].x - ThisX) * (vrp.node[tour[j]].x - ThisX);
            if (ThisDist <= CloseDist)
            {
                ThisDist += (vrp.node[tour[j]].y - ThisY) * (vrp.node[tour[j]].y - ThisY);
                if (ThisDist <= CloseDist)
                {
                    if (j < i)
                        break;
                    CloseDist = ThisDist;
                    ClosePt = j;
                }
            }
        }
        /*swapping tour[i] and tour[ClosePt]*/
        unsigned temp = tour[i];
        tour[i] = tour[ClosePt];
        tour[ClosePt] = temp;
    }
}

std::vector<std::vector<node_t>>
postprocess_tsp_approx(const VRP &vrp, std::vector<std::vector<node_t>> &solRoutes)
{
    std::vector<std::vector<node_t>> modifiedRoutes;

    unsigned nroutes = solRoutes.size();
    for (unsigned i = 0; i < nroutes; ++i)
    {
        // postprocessing solRoutes[i]
        unsigned sz = solRoutes[i].size();
        std::vector<node_t> cities(sz + 1);
        std::vector<node_t> tour(sz + 1);

        for (unsigned j = 0; j < sz; ++j)
            cities[j] = solRoutes[i][j];

        cities[sz] = 0; // the last node is the depot.

        tsp_approx(vrp, cities, tour, sz + 1);

        // the first element of the tour is now the depot. So, ignore tour[0] and insert the rest into the vector.

        std::vector<node_t> curr_route;
        for (unsigned kk = 1; kk < sz + 1; ++kk)
        {
            curr_route.push_back(tour[kk]);
        }

        modifiedRoutes.push_back(curr_route);
    }
    return modifiedRoutes;
}

void tsp_2opt(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, unsigned ncities)
{
    // 'cities' contains the original solution. It is updated during the course of the 2opt-scheme to contain the 2opt soln.
    // 'tour' is an auxillary array.

    // repeat until no improvement is made
    unsigned improve = 0;

    while (improve < 2)
    {
        double best_distance = 0.0;
        //~ best_distance += L2_dist(points.x_coords[cities[0]], points.y_coords[cities[0]], 0, 0); // computing distance of the first point in the route with the depot.
        best_distance += vrp.get_dist(DEPOT, cities[0]); // computing distance of the first point in the route with the depot.

        for (unsigned jj = 1; jj < ncities; ++jj)
        {
            //~ best_distance += L2_dist(points.x_coords[cities[jj-1]], points.y_coords[cities[jj-1]], points.x_coords[cities[jj]], points.y_coords[cities[jj]]);
            best_distance += vrp.get_dist(cities[jj - 1], cities[jj]);
        }
        //~ best_distance += L2_dist(points.x_coords[cities[ncities-1]], points.y_coords[cities[ncities-1]], 0, 0); // computing distance of the last point in the route with the depot.
        best_distance += vrp.get_dist(DEPOT, cities[ncities - 1]);
        // 1x 2x 3x 4 5
        //  1 2  3  4 5
        for (unsigned i = 0; i < ncities - 1; i++)
        {
            for (unsigned k = i + 1; k < ncities; k++)
            {
                for (unsigned c = 0; c < i; ++c)
                {
                    tour[c] = cities[c];
                }

                unsigned dec = 0;
                for (unsigned c = i; c < k + 1; ++c)
                {
                    tour[c] = cities[k - dec];
                    dec++;
                }

                for (unsigned c = k + 1; c < ncities; ++c)
                {
                    tour[c] = cities[c];
                }
                double new_distance = 0.0;
                //~ new_distance += L2_dist(points.x_coords[tour[0]], points.y_coords[tour[0]], 0, 0); // computing distance of the first point in the route with the depot.
                new_distance += vrp.get_dist(DEPOT, tour[0]);
                for (unsigned jj = 1; jj < ncities; ++jj)
                {
                    //~ new_distance += L2_dist(points.x_coords[tour[jj-1]], points.y_coords[tour[jj-1]], points.x_coords[tour[jj]], points.y_coords[tour[jj]]);
                    new_distance += vrp.get_dist(tour[jj - 1], tour[jj]);
                }
                //~ new_distance += L2_dist(points.x_coords[tour[ncities-1]], points.y_coords[tour[ncities-1]], 0, 0); // computing distance of the last point in the route with the depot.
                new_distance += vrp.get_dist(DEPOT, tour[ncities - 1]);

                if (new_distance < best_distance)
                {
                    // Improvement found so reset
                    improve = 0;
                    for (unsigned jj = 0; jj < ncities; jj++)
                        cities[jj] = tour[jj];
                    best_distance = new_distance;
                }
            }
        }
        improve++;
    }
}

std::vector<std::vector<node_t>>
postprocess_2OPT(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes)
{
    std::vector<std::vector<node_t>> postprocessed_final_routes;

    unsigned nroutes = final_routes.size();
    for (unsigned i = 0; i < nroutes; ++i)
    {
        // postprocessing final_routes[i]
        unsigned sz = final_routes[i].size();
        //~ unsigned* cities = (unsigned*) malloc(sizeof(unsigned) * (sz));
        //~ unsigned* tour = (unsigned*) malloc(sizeof(unsigned) * (sz));  // this is an auxillary array

        std::vector<node_t> cities(sz);
        std::vector<node_t> tour(sz);

        for (unsigned j = 0; j < sz; ++j)
            cities[j] = final_routes[i][j];

        std::vector<node_t> curr_route;

        if (sz > 2)                          // for sz <= 1, the cost of the path cannot change. So no point running this.
            tsp_2opt(vrp, cities, tour, sz); // MAIN

        for (unsigned kk = 0; kk < sz; ++kk)
        {
            curr_route.push_back(cities[kk]);
        }

        postprocessed_final_routes.push_back(curr_route);
    }
    return postprocessed_final_routes;
}

weight_t get_total_cost_of_routes(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes)
{
    weight_t total_cost = 0.0;
    for (unsigned ii = 0; ii < final_routes.size(); ++ii)
    {
        weight_t curr_route_cost = 0;
        //~ curr_route_cost += L2_dist(points.x_coords[final_routes[ii][0]], points.y_coords[final_routes[ii][0]], 0, 0); // computing distance of the first point in the route with the depot.
        curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
        for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
        {
            //~ curr_route_cost += L2_dist(points.x_coords[final_routes[ii][jj-1]], points.y_coords[final_routes[ii][jj-1]], points.x_coords[final_routes[ii][jj]], points.y_coords[final_routes[ii][jj]]);
            curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
        }
        //~ curr_route_cost += L2_dist(points.x_coords[final_routes[ii][final_routes[ii].size()-1]], points.y_coords[final_routes[ii][final_routes[ii].size()-1]], 0, 0); // computing distance of the last point in the route with the depot.
        curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);

        total_cost += curr_route_cost;
    }

    return total_cost;
}

//
// MAIN POST PROCESS ROUTINE
//
std::vector<std::vector<node_t>>
postProcessIt(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes, weight_t &minCost)
{
    std::vector<std::vector<node_t>> postprocessed_final_routes;

    auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
    auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
    auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);

//~ weight_t postprocessed_final_routes_cost;
#pragma omp parallel for
    for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz)
    {
        // include the better route between postprocessed_final_routes2[zzz] and postprocessed_final_routes3[zzz] in the final solution.

        std::vector<node_t> postprocessed_route2 = postprocessed_final_routes2[zzz];
        std::vector<node_t> postprocessed_route3 = postprocessed_final_routes3[zzz];

        unsigned sz2 = postprocessed_route2.size();
        unsigned sz3 = postprocessed_route3.size();

        // finding the cost of postprocessed_route2

        weight_t postprocessed_route2_cost = 0.0;
        //~ postprocessed_route2_cost += L2_dist(points.x_coords[postprocessed_route2[0]], points.y_coords[postprocessed_route2[0]], 0, 0); // computing distance of the first point in the route with the depot.
        postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]); // computing distance of the first point in the route with the depot.
        for (unsigned jj = 1; jj < sz2; ++jj)
        {
            //~ postprocessed_route2_cost += L2_dist(points.x_coords[postprocessed_route2[jj-1]], points.y_coords[postprocessed_route2[jj-1]], points.x_coords[postprocessed_route2[jj]], points.y_coords[postprocessed_route2[jj]]);
            postprocessed_route2_cost += vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
        }
        //~ postprocessed_route2_cost += L2_dist(points.x_coords[postprocessed_route2[sz2-1]], points.y_coords[postprocessed_route2[sz2-1]], 0, 0); // computing distance of the last point in the route with the depot.
        postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[sz2 - 1]);

        // finding the cost of postprocessed_route3

        weight_t postprocessed_route3_cost = 0.0;
        //~ postprocessed_route3_cost += L2_dist(points.x_coords[postprocessed_route3[0]], points.y_coords[postprocessed_route3[0]], 0, 0); // computing distance of the first point in the route with the depot.
        postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
        for (unsigned jj = 1; jj < sz3; ++jj)
        {
            //~ postprocessed_route3_cost += L2_dist(points.x_coords[postprocessed_route3[jj-1]], points.y_coords[postprocessed_route3[jj-1]], points.x_coords[postprocessed_route3[jj]], points.y_coords[postprocessed_route3[jj]]);
            postprocessed_route3_cost += vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
        }
        //~ postprocessed_route3_cost += L2_dist(points.x_coords[postprocessed_route3[sz3-1]], points.y_coords[postprocessed_route3[sz3-1]], 0, 0); // computing distance of the last point in the route with the depot.
        postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[sz3 - 1]);

        // postprocessed_route2_cost is lower
        if (postprocessed_route3_cost > postprocessed_route2_cost)
        {
            postprocessed_final_routes.push_back(postprocessed_route2);
        }
        // postprocessed_route3_cost is lower
        else
        {
            postprocessed_final_routes.push_back(postprocessed_route3);
        }
    }

    auto postprocessed_final_routes_cost = get_total_cost_of_routes(vrp, postprocessed_final_routes);

    minCost = postprocessed_final_routes_cost;

    return postprocessed_final_routes;
}

std::string get_base_name(const std::string &path)
{
    // Find the last path separator (works for both / and \)
    size_t last_slash_idx = path.find_last_of("/\\");
    std::string filename = (last_slash_idx == std::string::npos) ? path : path.substr(last_slash_idx + 1);

    // Find the last period (to remove extension)
    size_t period_idx = filename.rfind('.');
    if (period_idx != std::string::npos && period_idx != 0)
    { // Avoid cases like ".hiddenfile"
        filename = filename.substr(0, period_idx);
    }

    return filename;
}

// Writes routes and cost to a .routes file
bool writeRoutes(const std::string &filename, const std::vector<std::vector<int>> &routes, double cost)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return false;
    }

    // Write each route
    for (size_t i = 0; i < routes.size(); ++i)
    {
        file << "Route #" << (i + 1) << ":";
        for (int node : routes[i])
        {
            file << " " << node;
        }
        file << "\n";
    }

    // Write the cost (using fixed formatting in case it's a floating point number)
    // You can remove std::fixed and std::setprecision if you only want raw integers
    file << "Cost " << std::fixed << std::setprecision(2) << cost << "\n";

    file.close();
    return true;
}

void get_nearest_neighbours(
    VRP &vrp,
    std::vector<node_t> &edge_X,
    std::vector<node_t> &edge_Y,
    std::vector<weight_t> &edge_W)
{
    // -------------------- Basic Parameters --------------------
    const int num_nodes = vrp.getSize();
    const int num_neighbors = VRP::K;

    // -------------------- Extract Coordinates --------------------
    std::vector<double> x_coords(num_nodes);
    std::vector<double> y_coords(num_nodes);

    for (int i = 0; i < num_nodes; ++i)
    {
        x_coords[i] = vrp.node[i].x;
        y_coords[i] = vrp.node[i].y;
    }

    // -------------------- Build KD-Tree --------------------
    cobra::KDTree kd_tree(x_coords, y_coords);

    // -------------------- Compute Neighbor Edges --------------------
    int edge_counter = 0;
    const int max_edges = static_cast<int>(edge_X.size());

    for (int i = 1; i < num_nodes; ++i) // skip depot (assumed 0)
    {
        const weight_t dist_i_depot = vrp.get_dist(DEPOT, i);

        // +1 because the node itself may be included
        const auto nearest_nodes =
            kd_tree.GetNearestNeighbors(x_coords[i], y_coords[i], num_neighbors + 1);

        for (const int j : nearest_nodes)
        {
            // Skip invalid/self/depot nodes
            if (j == i || j == DEPOT)
                continue;

            // Safety check to avoid overflow (defensive programming)
            if (edge_counter >= max_edges)
                break;

            const weight_t saving =
                dist_i_depot +
                vrp.get_dist(DEPOT, j) -
                vrp.get_dist(i, j);

            edge_X[edge_counter] = i;
            edge_Y[edge_counter] = j;
            edge_W[edge_counter] = saving;

            ++edge_counter;
        }
    }
}

int main(int argc, char *argv[])
{
    try
    {
        // -------------------- Argument Validation --------------------
        if (argc < 2)
        {
            std::cerr << "Usage: " << argv[0] << " <filename.vrp> [options]\n"
                      << "Options:\n"
                      << "  -round <0|1>   Enable rounding default is 1\n"
                      << "  -K <int>       Number of nearest neighbors, default is 1500\n"
                      << "  -PR <0|1>      Print routes to file, default is 1\n";
            return EXIT_FAILURE;
        }

        // -------------------- Problem Initialization --------------------
        VRP vrp;
        vrp.read(argv[1]);

        // Default parameters
        VRP::isRound = true;
        VRP::K = 1500;
        bool print_routes = true;

        // -------------------- Argument Parsing --------------------
        for (int i = 2; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "-round" && i + 1 < argc)
            {
                VRP::isRound = (std::stoi(argv[++i]) == 1);
            }
            else if (arg == "-K" && i + 1 < argc)
            {
                int k_val = std::stoi(argv[++i]);
                VRP::K = std::min(k_val, static_cast<int>(vrp.getSize()) - 2);
            }
            else if (arg == "-PR" && i + 1 < argc)
            {
                print_routes = (std::stoi(argv[++i]) == 1);
            }
            else
            {
                std::cerr << "Warning: Unknown or incomplete argument '" << arg << "' ignored.\n";
            }
        }

        std::cout << "Problem loaded: " << argv[1] << std::endl;
        std::cout << "Rounding enabled: " << (VRP::isRound ? "Yes" : "No") << std::endl;
        std::cout << "Number of nearest neighbors (K): " << VRP::K << std::endl;
        std::cout << "Print routes to file: " << (print_routes ? "Yes" : "No") << std::endl;

        // -------------------- Memory Allocation --------------------
        const int num_nodes = vrp.getSize();
        const int num_edges = (num_nodes - 1) * VRP::K;

        std::vector<node_t> edge_X(num_edges), edge_Y(num_edges);
        std::vector<weight_t> edge_W(num_edges);

        // -------------------- Nearest Neighbor Computation --------------------
        auto t0 = std::chrono::high_resolution_clock::now();
        get_nearest_neighbours(vrp, edge_X, edge_Y, edge_W);
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> nn_time = t1 - t0;
        std::cout << "Time taken to compute neighbour list: "
                  << nn_time.count() << " seconds\n";

        // -------------------- Clarke & Wright Algorithm --------------------
        auto cw_start = std::chrono::high_resolution_clock::now();
        auto routes = RCPW(vrp, edge_X, edge_Y, edge_W);
        auto cw_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> cw_time = cw_end - cw_start;

        weight_t total_cost = calCost(vrp, routes);

        std::cout << "\n--- Parallel Clarke & Wright Savings Algorithm ---\n";
        std::cout << "Problem File: " << argv[1] << "\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Before preprocessing Solution Cost: " << total_cost << "\n";

        // -------------------- Local Search / Post-processing --------------------
        auto ls_start = std::chrono::high_resolution_clock::now();
        routes = postProcessIt(vrp, routes, total_cost);
        auto ls_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> ls_time = ls_end - ls_start;

        // -------------------- Validation & Final Metrics --------------------
        bool is_valid = verify_sol(vrp, routes, vrp.getCapacity());
        total_cost = calCost(vrp, routes);

        double total_time = nn_time.count() + cw_time.count() + ls_time.count();

        std::cout << "Total Solution Cost: " << total_cost << "\n";
        std::cout << "Number of Routes:   " << routes.size() << "\n";
        std::cout << "Clarke & Wright Time: " << cw_time.count() << " seconds\n";
        std::cout << "Local Search Time:    " << ls_time.count() << " seconds\n";
        std::cout << "Total Time Taken:     " << total_time << " seconds\n";
        std::cout << "Solution Validity:    " << (is_valid ? "VALID" : "INVALID") << "\n";
        std::cout << "--------------------------------------------------\n";

        // -------------------- Output Routes --------------------
        /*DEPOT is denoted with 0. subsequently the customers numbering will be start from 1 and goes to (Number of nodes - 1)
          For example, if there are 5 customers, then the depot is 0 and the customers are numbered from 1 to 5. The routes will be printed in the format:
           Route #1: 1 3 5
           Route #2: 2 4
           Cost 123.45
        */
        if (print_routes)
        {
            std::string base_name = get_base_name(argv[1]);
            std::string output_file = base_name + ".routes";

            if (writeRoutes(output_file, routes, total_cost))
            {
                std::cout << "Routes successfully written to " << output_file << "\n";
            }
            else
            {
                std::cerr << "Failed to write routes to file.\n";
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::string &e)
    {
        std::cerr << "CRITICAL ERROR: " << e << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}