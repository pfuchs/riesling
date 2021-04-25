#pragma once

#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

/* Conjugate gradients as described in K. P. Pruessmann, M. Weiger, M. B. Scheidegger, and P.
 * Boesiger, ‘SENSE: Sensitivity encoding for fast MRI’, Magnetic Resonance in Medicine, vol. 42,
 * no. 5, pp. 952–962, 1999.
 */
template <int Dim>
struct CG
{
  using X = Eigen::Tensor<Cx, Dim>;
  using SysFunc = std::function<void(X const &x, X &y)>;

  static void Run(SysFunc const &sys, float const &thresh, long const &max_its, X &img, Log &log)
  {
    log.info(FMT_STRING("Starting Conjugate Gradients, threshold {}"), thresh);
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    auto const dims = img.dimensions();
    X b(dims);
    X q(dims);
    X p(dims);
    X r(dims);
    b.setZero();
    q.setZero();
    p = img;
    r = img;
    float r_old = Norm2(r);
    float const a2 = Norm2(img);

    for (long icg = 0; icg < max_its; icg++) {
      sys(p, q);
      float const alpha = r_old / std::real(Dot(p, q));
      b.device(dev) = b + p * p.constant(alpha);
      r.device(dev) = r - q * q.constant(alpha);
      float const r_new = Norm2(r);
      float const beta = r_new / r_old;
      p.device(dev) = r + p * p.constant(beta);
      float const delta = r_new / a2;
      log.image(b, fmt::format(FMT_STRING("cg-b-{:02}.nii"), icg));
      log.info(FMT_STRING("CG {}: ɑ {} β {} δ {}"), icg, alpha, beta, delta);
      if (delta < thresh) {
        log.info(FMT_STRING("Reached convergence threshold"));
        break;
      }
      r_old = r_new;
    }
    img = b;
  }
};
