#pragma once

#include "log.h"
#include "op/recon.hpp"
#include "types.h"

/* F. Knoll, K. Bredies, T. Pock, and R. Stollberger, ‘Second order total generalized variation
 * (TGV) for MRI’, Magnetic Resonance in Medicine, vol. 65, no. 2, pp. 480–491, Feb. 2011,
 * doi: 10.1002/mrm.22595.
 */
Cx4 tgv(
  Index const max_its,
  float const thresh,
  float const alpha0,
  float const reduction,
  float const step_size,
  ReconOp &op,
  Cx3 const &radial,
  Log &log);
