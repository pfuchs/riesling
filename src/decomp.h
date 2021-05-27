#pragma once

#include "log.h"
#include "types.h"

Cx5 LowRank(Cx5 const &m, long const nRetain, Log const &log);
Cx2 Covariance(Cx2 const &data);
void PCA(Cx2 const &gram, Cx2 &vecs, R1 &vals);