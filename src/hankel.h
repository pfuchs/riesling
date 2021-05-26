#pragma once

#include "log.h"
#include "types.h"

Cx5 ToKernels(Cx4 const &grid, long const kSz, long const calSz, long const gapSz, Log &log);
void FromKernels(long const blkSz, long const kSz, Cx2 const &kernels, Cx4 &grid, Log &log);
