#pragma once

#include "info.h"
#include "kernel.h"
#include "log.h"

/*!
 * Parallel INfilling of Tomographic g Radial Inherent Sensitivities
 */
void PINOT(Info const &info, R3 const &traj, float const os, Kernel *const kb, Cx3 &data, Log &log);
