#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace nuclear_mpm {

struct ParallelConfig {
    bool enabled = false;
    std::size_t max_workers = 4;
    std::size_t scratch_memory_budget_bytes = 64U * 1024U * 1024U;
};

inline auto validate_parallel_config(const ParallelConfig &config) -> void {
    if (config.max_workers == 0) { throw std::invalid_argument("parallel max_workers must be positive"); }
    if (config.scratch_memory_budget_bytes == 0) {
        throw std::invalid_argument("parallel scratch_memory_budget_bytes must be positive");
    }
}

[[nodiscard]] inline auto select_parallel_worker_count(std::size_t grid_bytes, std::size_t requested_workers,
                                                        std::size_t budget_bytes) -> std::size_t {
    if (grid_bytes == 0) { throw std::invalid_argument("parallel grid_bytes must be positive"); }
    if (requested_workers == 0) { throw std::invalid_argument("parallel requested_workers must be positive"); }
    if (budget_bytes == 0) { throw std::invalid_argument("parallel budget_bytes must be positive"); }

    const std::size_t budgeted_workers = budget_bytes / grid_bytes;
    if (budgeted_workers < 2) { return 1; }
    return std::max<std::size_t>(1, std::min(requested_workers, budgeted_workers));
}

} // namespace nuclear_mpm
