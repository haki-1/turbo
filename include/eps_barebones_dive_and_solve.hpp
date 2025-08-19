// Copyright 2025 Hakan Hasan

#ifndef TURBO_EPS_BAREBONES_DIVE_AND_SOLVE_HPP
#define TURBO_EPS_BAREBONES_DIVE_AND_SOLVE_HPP

#include "common_solving.hpp"
#include "memory_gpu.hpp"
#include "lala/light_branch.hpp"
#include <mutex>
#include <thread>
#include <chrono>
#include <random>

#include "barebones_dive_and_solve.hpp"

namespace bt = ::battery;

#ifdef __CUDACC__

#include <cuda/std/chrono>
#include <cuda/semaphore>

#endif

#ifdef MULTI_GPU_WITH_MPI
#include "mpi.h"
#endif

namespace eps_barebones {

#ifdef __CUDACC__
#ifndef TURBO_IPC_ABSTRACT_DOMAIN
#ifdef MULTI_GPU_WITH_MPI

// Use without providing local_rank for the counter, and with - for the best_obj
void rma_window_create(MPI_Comm comm, MPI_Win *counter_win, int init_value, int local_rank = 1) {
  int size, rank, lnum = 0, *counter_mem = 0;
  MPI_Aint counter_size;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0 || local_rank == 0) {
    ++lnum;
  }
  counter_size = lnum * sizeof(int);
  if (counter_size > 0) {
    MPI_Alloc_mem(counter_size, MPI_INFO_NULL, &counter_mem);
    for (int i = 0; i < lnum; ++i) {
      counter_mem[i] = init_value;
    }
  }

  MPI_Win_create(counter_mem, counter_size, sizeof(int), MPI_INFO_NULL, comm, counter_win);
}

// Use with leader_rank = 0 for the counter, and with leader_rank = node_leader for the best_obj
int rma_window_nxtval(MPI_Win counter_win, int *value, MPI_Op op, int op_value, int leader_rank = 0) {
  const int one = op_value;
  int lrank = leader_rank;
  MPI_Aint lidx = 0;

  /** Update and return the counter */
  MPI_Win_lock(MPI_LOCK_SHARED, lrank, 0, counter_win);
  MPI_Fetch_and_op(&one, value, MPI_INT, lrank, lidx, op, counter_win);
  MPI_Win_unlock(lrank, counter_win);
  return 0;
}

// Simple gossip protocol logic for inter-node communication
int rma_window_crdt(MPI_Win counter_win, int *value, MPI_Op op, int op_value, const std::vector<int>& other_leaders, std::mt19937& gen) {
  if (other_leaders.empty()) {
    // printf("%%%%%% Apparantly only one node !\n");
    return -1;
  }
  std::uniform_int_distribution<> distrib(0, other_leaders.size() - 1);
  int random_leader = other_leaders[distrib(gen)];
  // printf("%%%%%% Selected index: %d\n", random_leader);

  return rma_window_nxtval(counter_win, value, op, op_value, random_leader);
}

void set_subproblem_vars(CP<Itv>& subproblem, std::vector<int>& vec, std::vector<int>& split_var_idxs, int idx_to_solve, int best_found) {
  using value_type = typename Itv::LB::value_type;
  for (int i = 0; i < vec.size() - 1; ++i) {
    Itv var_value = Itv((*subproblem.store)[split_var_idxs[i]].lb().value() + value_type{idx_to_solve / vec[i] % ((*subproblem.store)[split_var_idxs[i]].width().lb().value() + 1)});
    subproblem.store->embed(split_var_idxs[i], var_value);
  }

  if (subproblem.bab->is_minimization() && best_found != INT_MAX) {
    Itv obj_itv = Itv((*subproblem.store)[subproblem.bab->objective_var()].lb().value(), value_type{best_found - 1});
    subproblem.store->embed(subproblem.bab->objective_var(), obj_itv);
  }
  else if (subproblem.bab->is_maximization() && best_found != INT_MAX) {
    Itv obj_itv = Itv(value_type{-best_found + 1}, (*subproblem.store)[subproblem.bab->objective_var()].ub().value());
    subproblem.store->embed(subproblem.bab->objective_var(), obj_itv);
  }
}

int get_local_best_cp(CP<Itv>& global, CP<Itv>& subproblem) {
  if (global.bab->is_satisfaction()) {
    return -1;
  }
  if (!subproblem.best->is_top()) { // or with width of this Itv or top on the obj var only
    int best = (*subproblem.best)[subproblem.bab->objective_var()].lb().value();    
    return subproblem.bab->is_minimization() ? best : -best;
  }
  return INT_MAX;
  // return global.bab->is_minimization() ? INT_MAX : INT_MIN;
}

struct stats_info {
  unsigned long nodes;
  unsigned long fails;
  unsigned long solutions;
  unsigned long depth_max;
  unsigned long exhaustive;
  // unsigned long eps_num_subproblems; // here nothing to be aggregated - think again.....
  unsigned long eps_solved_subproblems;
  unsigned long eps_skipped_subproblems;
  unsigned long num_blocks_done;
  unsigned long fixpoint_iterations;
  unsigned long num_deductions;
  long timers[static_cast<int>(Timer::NUM_TIMERS)];  
};

void stats_sum(void *inP, void *inoutP, int *len, MPI_Datatype *dptr) {
  stats_info *in = (stats_info *)inP, *inout = (stats_info *)inoutP;

  for (int i = 0; i < *len; ++i) {
    inout[i].nodes += in[i].nodes;
    inout[i].fails += in[i].fails;
    inout[i].solutions += in[i].solutions;
    inout[i].depth_max = battery::max(inout[i].depth_max, in[i].depth_max);
    inout[i].exhaustive = inout[i].exhaustive && in[i].exhaustive;
    inout[i].eps_solved_subproblems += in[i].eps_solved_subproblems;
    inout[i].eps_skipped_subproblems += in[i].eps_skipped_subproblems;
    inout[i].num_blocks_done += in[i].num_blocks_done;
    inout[i].fixpoint_iterations += in[i].fixpoint_iterations;
    inout[i].num_deductions += in[i].num_deductions;
    for (int j = 0; j < static_cast<int>(Timer::NUM_TIMERS); ++j) {
      // Logic: FIRST_BLOCK_IDLE refers to the first idle block in the last subprolem solved by each GPU
      if (j == static_cast<int>(Timer::FIRST_BLOCK_IDLE)) {
        inout[i].timers[j] = battery::min(inout[i].timers[j], in[i].timers[j]);
      }
      else {
        inout[i].timers[j] += in[i].timers[j];
      }
    }    
  }
}

Statistics<CP<Itv>::basic_allocator_type> aggregate_stats(CP<Itv>& l_b_cp, int best_sol_root) {
  int blockcounts[2] = {10, static_cast<int>(Timer::NUM_TIMERS)};
  MPI_Aint displacements[2] = {
    offsetof(struct stats_info, nodes),
    offsetof(struct stats_info, timers)
  };
  MPI_Datatype types[2] = {MPI_UNSIGNED_LONG, MPI_LONG};
  
  MPI_Datatype mpi_stats_info_type;
  MPI_Type_create_struct(2, blockcounts, displacements, types, &mpi_stats_info_type);
  MPI_Type_commit(&mpi_stats_info_type);

  int rank, size;
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  stats_info stat_i {
    l_b_cp.stats.nodes, l_b_cp.stats.fails, l_b_cp.stats.solutions, l_b_cp.stats.depth_max, l_b_cp.stats.exhaustive ? 1UL : 0UL, l_b_cp.stats.eps_solved_subproblems,
    l_b_cp.stats.eps_skipped_subproblems, l_b_cp.stats.num_blocks_done, l_b_cp.stats.fixpoint_iterations, l_b_cp.stats.num_deductions
  };
  for(int i = 0; i < static_cast<int>(Timer::NUM_TIMERS); ++i) {
    stat_i.timers[i] = l_b_cp.stats.timers.time_of((Timer)i);
  }
  stat_i.timers[static_cast<int>(Timer::LATEST_BEST_OBJ_FOUND)] = 0;
  stats_info stat_i_recv = {};
  if (rank == best_sol_root) {
    stat_i = {};
    stat_i.exhaustive = l_b_cp.stats.exhaustive ? 1UL : 0UL;
    stat_i.timers[static_cast<int>(Timer::FIRST_BLOCK_IDLE)] = l_b_cp.stats.timers.time_of((Timer)static_cast<int>(Timer::FIRST_BLOCK_IDLE));
  }
    
  MPI_Op myOp;
  MPI_Op_create(stats_sum, 1, &myOp);
  // printf("Rank %d: Hey here are my stats before Reduction: node=%d, fails=%d, sol=%d, depth_max=%d\n\n", rank, stat_i.nodes, stat_i.fails, stat_i.solutions, stat_i.depth_max);
  // printf("Rank %d: Hey here are my stats before Reduction: node=%d\n\n", rank, stat_i.nodes);
  MPI_Reduce(&stat_i, &stat_i_recv, 1, mpi_stats_info_type, myOp, best_sol_root, MPI_COMM_WORLD);
  // printf("Rank %d: Hey we're just AFTER the reduction. Who is best_sol_root=%d\n", rank, best_sol_root);
  
  Statistics<CP<Itv>::basic_allocator_type> other_procs_stats(l_b_cp.stats.variables, l_b_cp.stats.constraints, l_b_cp.stats.optimization, l_b_cp.stats.print_statistics);
  other_procs_stats.nodes = stat_i_recv.nodes;
  other_procs_stats.fails = stat_i_recv.fails;
  other_procs_stats.solutions = stat_i_recv.solutions;
  other_procs_stats.depth_max = stat_i_recv.depth_max;
  other_procs_stats.exhaustive = (stat_i_recv.exhaustive != 0);
  // other_procs_stats.eps_num_subproblems = stat_i_recv.eps_num_subproblems;
  other_procs_stats.eps_solved_subproblems = stat_i_recv.eps_solved_subproblems;
  other_procs_stats.eps_skipped_subproblems = stat_i_recv.eps_skipped_subproblems;
  other_procs_stats.num_blocks_done = stat_i_recv.num_blocks_done;
  other_procs_stats.fixpoint_iterations = stat_i_recv.fixpoint_iterations;
  other_procs_stats.num_deductions = stat_i_recv.num_deductions;

  for(int i = 0; i < static_cast<int>(Timer::NUM_TIMERS); ++i) {
    other_procs_stats.timers.time_of((Timer)i) = stat_i_recv.timers[i];
  }
  // printf("Rank %d: Hey here are my stats AFTER Reduction: node=%d and the raw=%d\n\n", rank, other_procs_stats.nodes, stat_i_recv.nodes);
  // printf("Rank %d: Who is best_sol_root=%d\n", rank, best_sol_root);

  MPI_Op_free(&myOp);
  MPI_Type_free(&mpi_stats_info_type);

  return other_procs_stats;
}

void simple_subproblem_generation(CP<Itv>& cp, size_t num_subproblems_req, std::vector<int>& vec, std::vector<int>& split_var_idxs) {
  int A_k = 1;
  vec.push_back(A_k);

  CP<Itv> cp_copy(cp), cp_copy_prev(cp_copy);
  while (A_k < num_subproblems_req) {
    auto branches = cp_copy.split->split();
    if (branches.size() != 2) break;
    
    cp_copy.iprop->deduce(branches[0]);
    size_t idx = 0;
    while (idx < cp_copy.store->vars() && (*cp_copy.store)[idx].width().lb().value() == (*cp_copy_prev.store)[idx].width().lb().value()) {
      ++idx;
    }
    if (!split_var_idxs.empty() && idx == split_var_idxs.back())
    {
      cp_copy_prev.iprop->deduce(branches[0]);
      continue;
    }
    
    A_k *= (*cp_copy_prev.store)[idx].width().lb().value() + 1;
    split_var_idxs.push_back(idx);

    cp_copy_prev.iprop->deduce(branches[0]);
  }

  std::reverse(split_var_idxs.begin(), split_var_idxs.end());

  for (size_t i = 0; i < split_var_idxs.size(); ++i) {
    vec.push_back(vec.back() * ((*cp.store)[split_var_idxs[i]].width().lb().value() + 1));
  }
}

using namespace barebones;

template<class CP, class Timepoint>
bool wait_solving_ends_MPI(cuda::std::atomic_flag& stop, CP& root, const Timepoint& start, MPI_Win counterWin, MPI_Win bestWin, int& ud_CPU_bound, int& ud_GPU_bound, int node_leader_rank, const std::vector<int>& other_leaders, std::mt19937& gen) {
  int rank, size;
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  size_t c = 0;

  cudaEvent_t event;
  cudaEventCreateWithFlags(&event,cudaEventDisableTiming);
  cudaEventRecord(event);
  while(!must_quit(root) && check_timeout(root, start) && cudaEventQuery(event) == cudaErrorNotReady) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ++c;
    if (c % 4 == 0) {
      int best_value;
      if (root.bab->is_optimization()) {
        rma_window_nxtval(bestWin, &best_value, MPI_MIN, ud_GPU_bound, node_leader_rank);
        ud_CPU_bound = std::min(best_value, ud_CPU_bound);
      }
    }

    if (c % 12 == 0 && rank == node_leader_rank) {
      int best_value, my_offer;
      if (root.bab->is_optimization()) {
        rma_window_nxtval(bestWin, &my_offer, MPI_MIN, INT_MAX, node_leader_rank);
        if (rma_window_crdt(bestWin, &best_value, MPI_MIN, my_offer, other_leaders, gen) == 0) {
          rma_window_nxtval(bestWin, &my_offer, MPI_MIN, best_value, node_leader_rank);

          // printf("%%%%%%------L-rank=%d---------myyyy_ofer=%d\n", rank, my_offer);
          // printf("%%%%%%------L-rank=%d---------opost_ofer=%d\n", rank, best_value);
        }
      }
    }
    
    if (rank == node_leader_rank) {
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, counterWin);
      MPI_Win_unlock(0, counterWin);
      if (root.bab->is_optimization()) {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, bestWin);
        MPI_Win_unlock(0, bestWin);
      }
    }
  }
  if(cudaEventQuery(event) == cudaErrorNotReady) {
    stop.test_and_set();
    root.prune();
    return true;
  }
  else {
    cudaError error = cudaDeviceSynchronize();
    if(error == cudaErrorIllegalAddress) {
      printf("%% ERROR: CUDA kernel failed due to an illegal memory access. This might be due to a stack overflow because it is too small. Try increasing the stack size with the options -stack. If it does not work, please report it as a bug.\n");
      exit(EXIT_FAILURE);
    }
    CUDAEX(error);
    return false;
  }
}

void solve(CP<Itv>& cp, std::chrono::steady_clock::time_point start, MPI_Win counterWin, MPI_Win bestWin, int current_subproblem_idx, int node_leader_rank, const std::vector<int>& other_leaders, std::mt19937& gen) {
  bool ps = cp.config.print_statistics;
  if (current_subproblem_idx) {
    cp.stats.print_statistics = cp.config.print_statistics = false;
  }

  MemoryConfig mem_config = configure_gpu_barebones(cp);
  cp.stats.print_statistics = cp.config.print_statistics = ps;
  auto unified_data = bt::make_unique<UnifiedData, ConcurrentAllocator>(cp, mem_config);
  auto grid_data = bt::make_unique<bt::unique_ptr<GridData, bt::global_allocator>, ConcurrentAllocator>();
  initialize_global_data<<<1,1>>>(unified_data.get(), grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
  /** We wait that either the solving is interrupted, or that all threads have finished. */
  /** Block the signal CTRL-C to notify the threads if we must exit. */
  block_signal_ctrlc();
  gpu_barebones_solve
    <<<static_cast<unsigned int>(cp.stats.num_blocks),
      CUDA_THREADS_PER_BLOCK,
      mem_config.shared_bytes>>>
    (unified_data.get(), grid_data->get());
  auto now = std::chrono::steady_clock::now();
  int64_t time_to_kernel_start = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
  bool interrupted = wait_solving_ends_MPI(unified_data->stop, unified_data->root, start, counterWin, bestWin, unified_data->bound_CPU, unified_data->bound_GPU, node_leader_rank, other_leaders, gen);
  CUDAEX(cudaDeviceSynchronize());
  reduce_blocks<<<1,1>>>(unified_data.get(), grid_data->get());
  CUDAEX(cudaDeviceSynchronize());
  auto& uroot = unified_data->root;
  if(uroot.stats.solutions > 0) {
    // We add the time before the kernel starts to the time needed to find the best bound.
    uroot.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) += time_to_kernel_start;
    if(uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) != 0) {
      uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) += time_to_kernel_start;
    }
    // cp.print_solution(*uroot.best);
  }
  // uroot.stats.print_mzn_final_separator();
  // if(uroot.config.print_statistics) {
  //   uroot.config.print_mzn_statistics();
  //   uroot.stats.print_mzn_statistics(uroot.config.verbose_solving);
  //   if(uroot.bab->is_optimization() && uroot.stats.solutions > 0) {
  //     uroot.stats.print_mzn_objective(uroot.best->project(uroot.bab->objective_var()), uroot.bab->is_minimization());
  //   }
  //   unified_data->root.stats.print_mzn_end_stats();
  // }
  deallocate_global_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());

  // We add the time before the kernel starts even if not found solution.
  if(uroot.stats.solutions <= 0 && uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) != 0) {
    uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) += time_to_kernel_start;
  }
  if (cp.bab->is_satisfaction()) {
    uroot.best->extract(*cp.best);
  }

  cp.meet(uroot);
}

void print_container(const std::vector<int>& c)
{
    for (int i : c)
        std::cout << i << ' ';
    std::cout << '\n';
}

void eps_barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config, int& argc, char**& argv) {
  int rank, size;
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  std::cout << "%%%%%%rank: " << rank << ", size: " << size << std::endl;

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  int count;
  cudaGetDeviceCount(&count);

  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  int local_rank, local_size, node_leader_rank;
  MPI_Comm_size( shmcomm, &local_size );
  MPI_Comm_rank( shmcomm, &local_rank );
  printf("%%%%%%Rank %d is running on %s; it sees %d GPUs in total and its local rank is %d out of local size: %d. \n", rank, hostname, count, local_rank, local_size);
  CUDAEX(cudaSetDevice(local_rank));
  if (local_rank == 0) {
    node_leader_rank = rank;
  }
  MPI_Bcast(&node_leader_rank, 1, MPI_INT, 0, shmcomm);
  MPI_Comm_free(&shmcomm);

  // Find all leader ranks (and remove self from the local copy of the vec)
  int local_ranks_arr[size];
  MPI_Allgather(&local_rank, 1, MPI_INT, local_ranks_arr, 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> peers_ranks;
  for (int i = 0; i < size; ++i) {
    if (local_ranks_arr[i] == 0) {
      peers_ranks.push_back(i);
    }
  }
  auto it = std::find(peers_ranks.begin(), peers_ranks.end(), rank);
  if (it == peers_ranks.end()) {
    peers_ranks.clear();
  }
  else {
    peers_ranks.erase(it);
  }
  printf("%%%%%%Rank %d: peers: ", rank);
  print_container(peers_ranks);

  std::random_device rd;  // a seed source for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()

  if(config.print_intermediate_solutions) {
    printf("%% WARNING: -arch multi_gpu_barebones is incompatible with -i and -a (it cannot print intermediate solutions).\n");
  }
  auto start = std::chrono::steady_clock::now();
  check_support_managed_memory();
  check_support_concurrent_managed_memory();
  /** We start with some preprocessing to reduce the number of variables and constraints. */
  CP<Itv> cp(config);
  if (rank != 0) { cp.stats.print_statistics = false; }
  cp.preprocess();
  if(cp.iprop->is_bot()) {
    if (rank == 0) {
      cp.print_final_solution();
      cp.print_mzn_statistics();
    }
    return;
  }

  /** Number of MPI subproblems. */
  if(cp.config.mpi_subproblems == -1) {
    cp.config.mpi_subproblems = 30 * size;
  }
  printf("%%%%%%cp.config.mpi_subproblems=%d\n", cp.config.mpi_subproblems);

  MPI_Win counter_win, best_win;
  rma_window_create(MPI_COMM_WORLD, &counter_win, 0);
  // We always seek to minimize.
  if (cp.bab->is_optimization()) {
    rma_window_create(MPI_COMM_WORLD, &best_win, INT_MAX, local_rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<int> vec, split_var_idxs;  
  simple_subproblem_generation(cp, cp.config.mpi_subproblems, vec, split_var_idxs);

  CP<Itv> local_best_cp_2(cp);

  int current_subproblem_idx, best_value;
  rma_window_nxtval(counter_win, &current_subproblem_idx, MPI_SUM, 1);
  printf("%%%%%% Rank %d: leader: %d, counter: %d\n", rank, node_leader_rank, current_subproblem_idx);
  if (cp.bab->is_optimization()) {
    rma_window_nxtval(best_win, &best_value, MPI_MIN, INT_MAX, node_leader_rank);
  }

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  MPI_Barrier(MPI_COMM_WORLD);

  int dev;
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);

  bool reset_num_blocks_done = false;
  int oks = 0; int bads = 0;

  while (current_subproblem_idx < vec.back() && (cp.config.timeout_ms == 0 || duration < cp.config.timeout_ms))
  {
    CP<Itv> subproblem(cp);
    set_subproblem_vars(subproblem, vec, split_var_idxs, current_subproblem_idx, best_value);

    GaussSeidelIteration fp_engine;
    fp_engine.fixpoint(subproblem.iprop->num_deductions(), [&](size_t i) { return subproblem.iprop->deduce(i); }
    , [&](){ return subproblem.iprop->is_bot(); }
    );

    if(!subproblem.iprop->is_bot()) {
      printf("%%%%%%Rank %d: starting subproblem index %d out of %d\n\n", rank, current_subproblem_idx, vec.back());
      ++oks;
      solve(subproblem, start, counter_win, best_win, current_subproblem_idx, node_leader_rank, peers_ranks, gen);

      local_best_cp_2.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) = 0;
      // we don't need to check if the objective is better because if it is not it won't be found because of restricton of the domain
      if(local_best_cp_2.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) < subproblem.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND)) {
        local_best_cp_2.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) = 0;
      }

      local_best_cp_2.meet(subproblem);
      if (cp.bab->is_satisfaction()) {
        subproblem.best->extract(*local_best_cp_2.best);
      }

      local_best_cp_2.stats.num_blocks = size * subproblem.stats.num_blocks;
      if (reset_num_blocks_done) local_best_cp_2.stats.num_blocks_done -= subproblem.stats.num_blocks;
      reset_num_blocks_done = true;      
    }
    else {
      printf("%%%%%%Rank %d: SKIPPING subproblem index %d out of %d, because it is not NDI\n\n", rank, current_subproblem_idx, vec.back());
      ++bads;
      local_best_cp_2.stats.eps_skipped_subproblems += local_best_cp_2.stats.eps_num_subproblems;
      if(!(!must_quit(subproblem) && check_timeout(subproblem, start))) local_best_cp_2.stats.exhaustive = false;
    }
        
    int best_sol_in_this_subprob = get_local_best_cp(cp, subproblem);

    printf("%%%%%%Rank %d: ended subproblem index %d out of __ and have explored %d number of nodes and found solutions %d\n\n", rank, current_subproblem_idx, subproblem.stats.nodes, subproblem.stats.solutions);    
    
    rma_window_nxtval(counter_win, &current_subproblem_idx, MPI_SUM, 1);
    if (cp.bab->is_optimization()) {
      rma_window_nxtval(best_win, &best_value, MPI_MIN, best_sol_in_this_subprob, node_leader_rank);
      best_value = std::min(best_value, best_sol_in_this_subprob);
    }

    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    CUDAEX(cudaDeviceReset());
  }
  printf("%%%%%%RANK %d: outside loop ?!\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
  int local_data_pair[2], global_data_pair[2];
  MPI_Op opReduce;
  if (cp.bab->is_satisfaction()) {
    local_data_pair[0] = local_best_cp_2.stats.solutions > 0 ? 1 : 0;
    opReduce = MPI_MAXLOC;
  }
  else if (cp.bab->is_minimization()) {
    local_data_pair[0] = INT_MAX;
    opReduce = MPI_MINLOC;
  }
  else if (cp.bab->is_maximization()) {
    local_data_pair[0] = INT_MIN;
    opReduce = MPI_MAXLOC;
  }
  if (!local_best_cp_2.best->is_top() && cp.bab->is_optimization()) {
    local_data_pair[0] = (*local_best_cp_2.best)[local_best_cp_2.bab->objective_var()].lb().value();
  }
  local_data_pair[1] = rank;
  // printf("%%%%%%Rank %d: Real (in end) local is %d, AND local_2 is %d;;;;; if(..._2) is %s\n\n", rank, local_data_pair[0], (*local_best_cp_2.best)[(*local_best_cp_2.bab).objective_var()].lb().value(), !(*local_best_cp_2.best).is_top() ? "true" : "false");
  MPI_Allreduce(&local_data_pair, &global_data_pair, 1, MPI_2INT, opReduce, MPI_COMM_WORLD);
    
  Statistics<CP<Itv>::basic_allocator_type> s = aggregate_stats(local_best_cp_2, global_data_pair[1]);

  if (rank == global_data_pair[1])
  {
    local_best_cp_2.stats.print_statistics = cp.config.print_statistics;

    if(local_best_cp_2.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) > s.timers.time_of(Timer::FIRST_BLOCK_IDLE)) {
      local_best_cp_2.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) = 0;
    }
    else {
      s.timers.time_of(Timer::FIRST_BLOCK_IDLE) = 0;
    }

    local_best_cp_2.stats.meet(s);
    bool tmp_exhaustive = local_best_cp_2.stats.exhaustive;
    // printf("%%%%%%is exhaustive: %s", local_best_cp_2.stats.exhaustive ? "true" : "false");
    check_timeout(local_best_cp_2, start);
    local_best_cp_2.stats.exhaustive = tmp_exhaustive;
    // local_best_cp_2.stats.eps_num_subproblems = vec.back(); // !!! THIS SHOULD BE ON THE OTHERE ONE BUT IS NEEEDED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    local_best_cp_2.stats.depth_max += vec.size() - 1;
    local_best_cp_2.stats.timers.time_of(Timer::PREPROCESSING) = cp.stats.timers.time_of(Timer::PREPROCESSING);
    local_best_cp_2.stats.eps_num_subproblems *= vec.back();

    if(local_best_cp_2.stats.solutions > 0) {
      local_best_cp_2.print_solution();
    }
    local_best_cp_2.stats.print_mzn_final_separator();
    if(local_best_cp_2.config.print_statistics) {
      local_best_cp_2.stats.print_stat("MPI_procs", (size_t)size);
      local_best_cp_2.stats.print_stat("MPI_eps_num_sub", (size_t)vec.back());
      // local_best_cp_2.stats.print_stat("MPI_oks_subs_found", oks);
      // local_best_cp_2.stats.print_stat("MPI_bad_subs_found", bads);
      local_best_cp_2.config.print_mzn_statistics();
      local_best_cp_2.stats.print_mzn_statistics(cp.config.verbose_solving);
      if(local_best_cp_2.bab->is_optimization() && local_best_cp_2.stats.solutions > 0) {
        local_best_cp_2.stats.print_mzn_objective(local_best_cp_2.best->project(local_best_cp_2.bab->objective_var()), local_best_cp_2.bab->is_minimization());
      }
      local_best_cp_2.stats.print_mzn_end_stats();
    }
    // local_best_cp_2.print_mzn_statistics();
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win_free(&counter_win);
  if (cp.bab->is_optimization()) {
    MPI_Win_free(&best_win);
  }

  MPI_Finalize();
}

#endif // MULTI_GPU_WITH_MPI
#endif // TURBO_IPC_ABSTRACT_DOMAIN
#endif // __CUDACC__

#if defined(TURBO_IPC_ABSTRACT_DOMAIN) || !defined(__CUDACC__) || !defined(MULTI_GPU_WITH_MPI)

void eps_barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config, int& argc, char**& argv) {
#ifdef TURBO_IPC_ABSTRACT_DOMAIN
  std::cerr << "-arch multi_gpu_barebones does not support IPC abstract domain." << std::endl;
#elif !defined(__CUDACC__)
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#else
  std::cerr << "To run Turbo in multi-GPU environment you need to build Turbo with the option MULTI_GPU_WITH_MPI." << std::endl;
#endif
}

#endif

} // namespace eps_barebones

#endif // TURBO_EPS_BAREBONES_DIVE_AND_SOLVE_HPP
