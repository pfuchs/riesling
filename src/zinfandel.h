#pragma once

#include "info.h"
#include "log.h"
#include "sense.h"

/*!
 * ZTE Infilling From Autocallibration NeighbourhooD ELements
 */
void zinfandel(
    long const gap_sz,
    long const n_src,
    long const n_cal_spoke,
    long const n_cal_read,
    float const lambda,
    Cx3 &ks,
    Log &log);

void zinfandel2(
    long const gap_sz,
    long const n_src,
    long const n_cal_read,
    float const lambda,
    Cx3 &ks,
    Log &log);

/*!
 *  Helper functions exposed for testing
 */
Cx2 GrabSources(
    Cx3 const &ks,
    float const scale,
    std::vector<long> const &src_offsets,
    long const st_cal_read,
    long const n_cal_read,
    std::vector<long> const &spokes);

Cx2 GrabTargets(
    Cx3 const &ks,
    float const scale,
    long const st_tgt_read,
    long const n_tgt_read,
    std::vector<long> const &spokes);
