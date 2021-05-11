#pragma once

#include "log.h"
#include "types.h"

void ToKernels(long const blkSz, long const kSz, Cx4 const &grid, Cx2 &kernels, Log &log);
void FromKernels(long const blkSz, long const kSz, Cx2 const &kernels, Cx4 &grid, Log &log);
