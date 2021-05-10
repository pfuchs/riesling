#pragma once
#include "info.h"
#include "kernel.h"
#include "log.h"
#include "types.h"

void MIRIN(
    Info const &info,
    R3 const &traj,
    float const os,
    Kernel *const kb,
    std::string const &sdc,
    long const calSz,
    long const kernelSz,
    long const its,
    long const retain,
    Cx3 &data,
    Log &log);
