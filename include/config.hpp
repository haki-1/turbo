// Copyright 2022 Pierre Talbot

#ifndef TURBO_CONFIG_HPP
#define TURBO_CONFIG_HPP

#include "battery/allocator.hpp"
#include "battery/string.hpp"
#include <cinttypes>
#include <optional>

#ifdef __CUDACC__
  #include <cuda.h>
#endif

enum class Arch {
  CPU,
  GPU,
  BAREBONES,
  HYBRID,
  MULTI_GPU_BAREBONES
};

enum class FixpointKind {
  AC1,
  WAC1
};

enum class InputFormat {
  XCSP3,
  FLATZINC
};

template<class Allocator>
struct Configuration {
  using allocator_type = Allocator;
  bool print_intermediate_solutions; // (only optimization problems).
  size_t stop_after_n_solutions; // 0 for all solutions (satisfaction problems only).
  size_t stop_after_n_nodes; // size_t MAX values for all nodes.
  bool free_search;
  bool print_statistics;
  int verbose_solving;
  bool print_ast;
  bool only_global_memory;
  bool force_ternarize;
  bool disable_simplify;
  bool network_analysis;
  size_t timeout_ms;
  size_t or_nodes;
  int subproblems_power;
  size_t subproblems_factor;
  int mpi_subproblems;
  size_t stack_kb;
  Arch arch;
  FixpointKind fixpoint;
  size_t wac1_threshold;
  size_t seed;
  battery::string<allocator_type> eps_var_order;
  battery::string<allocator_type> eps_value_order;
  battery::string<allocator_type> problem_path;
  battery::string<allocator_type> version;
  battery::string<allocator_type> hardware;

  CUDA Configuration(const allocator_type& alloc = allocator_type{}):
    print_intermediate_solutions(false),
    stop_after_n_solutions(1),
    stop_after_n_nodes(std::numeric_limits<size_t>::max()),
    free_search(false),
    verbose_solving(0),
    print_ast(false),
    print_statistics(false),
    only_global_memory(false),
    force_ternarize(false),
    disable_simplify(false),
    network_analysis(false),
    timeout_ms(0),
    or_nodes(0),
    subproblems_power(-1),
    subproblems_factor(30),
    mpi_subproblems(-1),
    stack_kb(
      #ifdef TURBO_IPC_ABSTRACT_DOMAIN
        32
      #else
        0
      #endif
    ),
    arch(
      #ifdef __CUDACC__
        Arch::BAREBONES
      #else
        Arch::CPU
      #endif
    ),
    fixpoint(
      #ifdef __CUDACC__
        FixpointKind::WAC1
      #else
        FixpointKind::AC1
      #endif
    ),
    wac1_threshold(0),
    seed(0),
    eps_value_order("default", alloc),
    eps_var_order("default", alloc),
    problem_path(alloc),
    version(alloc),
    hardware(alloc)
  {}

  Configuration(Configuration<allocator_type>&&) = default;
  Configuration(const Configuration<allocator_type>&) = default;

  template<class Alloc>
  CUDA Configuration(const Configuration<Alloc>& other, const allocator_type& alloc = allocator_type{}) :
    print_intermediate_solutions(other.print_intermediate_solutions),
    stop_after_n_solutions(other.stop_after_n_solutions),
    stop_after_n_nodes(other.stop_after_n_nodes),
    free_search(other.free_search),
    print_statistics(other.print_statistics),
    verbose_solving(other.verbose_solving),
    print_ast(other.print_ast),
    only_global_memory(other.only_global_memory),
    force_ternarize(other.force_ternarize),
    disable_simplify(other.disable_simplify),
    network_analysis(other.network_analysis),
    timeout_ms(other.timeout_ms),
    or_nodes(other.or_nodes),
    subproblems_power(other.subproblems_power),
    subproblems_factor(other.subproblems_factor),
    mpi_subproblems(other.mpi_subproblems),
    stack_kb(other.stack_kb),
    arch(other.arch),
    fixpoint(other.fixpoint),
    wac1_threshold(other.wac1_threshold),
    seed(other.seed),
    eps_var_order(other.eps_var_order, alloc),
    eps_value_order(other.eps_value_order, alloc),
    problem_path(other.problem_path, alloc),
    version(other.version, alloc),
    hardware(other.hardware, alloc)
  {}

  template <class Alloc2>
  CUDA Configuration<allocator_type>& operator=(const Configuration<Alloc2>& other) {
    print_intermediate_solutions = other.print_intermediate_solutions;
    stop_after_n_solutions = other.stop_after_n_solutions;
    stop_after_n_nodes = other.stop_after_n_nodes;
    free_search = other.free_search;
    verbose_solving = other.verbose_solving;
    print_ast = other.print_ast;
    print_statistics = other.print_statistics;
    only_global_memory = other.only_global_memory;
    force_ternarize = other.force_ternarize;
    disable_simplify = other.disable_simplify;
    network_analysis = other.network_analysis;
    timeout_ms = other.timeout_ms;
    or_nodes = other.or_nodes;
    subproblems_power = other.subproblems_power;
    subproblems_factor = other.subproblems_factor;
    mpi_subproblems = other.mpi_subproblems;
    stack_kb = other.stack_kb;
    arch = other.arch;
    fixpoint = other.fixpoint;
    wac1_threshold = other.wac1_threshold;
    seed = other.seed;
    eps_var_order = other.eps_var_order;
    eps_value_order = other.eps_value_order;
    problem_path = other.problem_path;
    version = other.version;
    hardware = other.hardware;
  }

  CUDA void print_commandline(const char* program_name) {
    printf("%s -t %" PRIu64 " %s-n %" PRIu64 " %s%s%s%s",
      program_name,
      timeout_ms,
      (print_intermediate_solutions ? "-a ": ""),
      stop_after_n_solutions,
      (print_intermediate_solutions ? "-i ": ""),
      (free_search ? "-f " : ""),
      (print_statistics ? "-s " : ""),
      (print_ast ? "-ast " : "")
    );
    for(int i = 0; i < verbose_solving; ++i) {
      printf("-v ");
    }
    if(arch != Arch::CPU) {
      printf("-arch %s -or %" PRIu64 " -sub %d -subfactor %" PRIu64 " -stack %" PRIu64 " ", name_of_arch(arch), or_nodes, subproblems_power, subproblems_factor, stack_kb);
      if (arch == Arch::MULTI_GPU_BAREBONES) { printf("-mpi_sub %d ", mpi_subproblems); }
      if(only_global_memory) { printf("-globalmem "); }
    }
    else {
      printf("-arch cpu -p %" PRIu64 " ", or_nodes);
    }
    if(disable_simplify) { printf("-disable_simplify "); }
    if(force_ternarize) { printf("-force_ternarize "); }
    if(network_analysis) { printf("-network_analysis "); }
    printf("-fp %s ", name_of_fixpoint(fixpoint));
    if(fixpoint == FixpointKind::WAC1) {
      printf("-wac1_threshold %" PRIu64 " ", wac1_threshold);
    }
    printf("-seed %" PRIu64 " ", seed);
    printf("-eps_var_order %s ", eps_var_order.data());
    printf("-eps_value_order %s ", eps_value_order.data());
    if(version.size() != 0) {
      printf("-version %s ", version.data());
    }
    if(hardware.size() != 0) {
      printf("-hardware \'%s\' ", hardware.data());
    }
    printf("-cutnodes %" PRIu64 " ", stop_after_n_nodes == std::numeric_limits<size_t>::max() ? 0 : stop_after_n_nodes);
    printf("%s", problem_path.data());
  }

  CUDA const char* name_of_fixpoint(FixpointKind fixpoint) const {
    switch(fixpoint) {
      case FixpointKind::AC1:
        return "ac1";
      case FixpointKind::WAC1:
        return "wac1";
      default:
        assert(0);
        return "Unknown";
    }
  }

  CUDA const char* name_of_arch(Arch arch) const {
    switch(arch) {
      case Arch::CPU:
        return "cpu";
      case Arch::GPU:
        return "gpu";
      case Arch::BAREBONES:
        return "barebones";
      case Arch::HYBRID:
        return "hybrid";
      case Arch::MULTI_GPU_BAREBONES:
        return "multi_gpu_barebones";
      default:
        assert(0);
        return "Unknown";
    }
  }

  CUDA void print_mzn_statistics() const {
    printf("%%%%%%mzn-stat: problem_path=\"%s\"\n", problem_path.data());
    printf("%%%%%%mzn-stat: solver=\"Turbo\"\n");
    printf("%%%%%%mzn-stat: version=\"%s\"\n", (version.size() == 0) ? "1.2.9" : version.data());
    printf("%%%%%%mzn-stat: hardware=\"%s\"\n", (hardware.size() == 0) ? "unspecified" : hardware.data());
    printf("%%%%%%mzn-stat: arch=\"%s\"\n", name_of_arch(arch));
    printf("%%%%%%mzn-stat: fixpoint=\"%s\"\n", name_of_fixpoint(fixpoint));
    // printf("%%%%%%mzn-stat: subproblems_power=\"%d\"\n", subproblems_power); // do not print because it must be printed before it is modified in barebones.
    printf("%%%%%%mzn-stat: subproblems_factor=\"%d\"\n", subproblems_factor);
    if(fixpoint == FixpointKind::WAC1) {
      printf("%%%%%%mzn-stat: wac1_threshold=%" PRIu64 "\n", wac1_threshold);
    }
    printf("%%%%%%mzn-stat: seed=%" PRIu64 "\n", seed);
    printf("%%%%%%mzn-stat: eps_var_order=\"%s\"\n", eps_var_order.data());
    printf("%%%%%%mzn-stat: eps_value_order=\"%s\"\n", eps_value_order.data());
    printf("%%%%%%mzn-stat: free_search=\"%s\"\n", free_search ? "yes" : "no");
    printf("%%%%%%mzn-stat: or_nodes=%" PRIu64 "\n", or_nodes);
    printf("%%%%%%mzn-stat: timeout_ms=%" PRIu64 "\n", timeout_ms);
    if(arch != Arch::CPU) {
      printf("%%%%%%mzn-stat: threads_per_block=%d\n", CUDA_THREADS_PER_BLOCK);
      printf("%%%%%%mzn-stat: stack_size=%" PRIu64 "\n", stack_kb * 1000);
      #ifdef CUDA_VERSION
        printf("%%%%%%mzn-stat: cuda_version=%d\n", CUDA_VERSION);
      #endif
      #ifdef __CUDA_ARCH__
        printf("%%%%%%mzn-stat: cuda_architecture=%d\n", __CUDA_ARCH__);
      #endif
    }
    printf("%%%%%%mzn-stat: cutnodes=%" PRIu64 "\n", stop_after_n_nodes == std::numeric_limits<size_t>::max() ? 0 : stop_after_n_nodes);
  }

  CUDA InputFormat input_format() const {
    if(problem_path.ends_with(".fzn")) {
      return InputFormat::FLATZINC;
    }
    else if(problem_path.ends_with(".xml")) {
      return InputFormat::XCSP3;
    }
    else {
      printf("ERROR: Unknown input format for the file %s [supported extension: .xml and .fzn].\n", problem_path.data());
      exit(EXIT_FAILURE);
    }
  }
};

void usage_and_exit(const std::string& program_name);
Configuration<battery::standard_allocator> parse_args(int argc, char** argv);

#endif
