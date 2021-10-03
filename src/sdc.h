#pragma once

#include "info.h"
#include "log.h"
#include "types.h"
#include <unordered_map>

// Forward declare
struct Trajectory;
struct GridOp;

namespace SDC {

void Load(
    std::string const &fname, Trajectory const &traj, std::unique_ptr<GridOp> &gridder, Log &log);
R2 Pipe(Trajectory const &traj, std::unique_ptr<GridOp> &gridder, Log &log);
R2 Radial(Trajectory const &traj, Log &log);
} // namespace SDC
