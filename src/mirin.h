#pragma once
#include "log.h"
#include "types.h"

void mirin(
    long const gap_sz,
    long const n_src,
    long const n_read1,
    long const n_spokes,
    long const its,
    float const frac,
    Cx3 &ks,
    Log &log);
