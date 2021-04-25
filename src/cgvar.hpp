#pragma once

#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

template <int Dim>
struct CGVar
{
  using X = Eigen::Tensor<Cx, Dim>;
  using SysFunc = std::function<void(X const &x, X &y, float const p)>;

  static void
  Run(SysFunc const &sys,
      float const &thresh,
      long const &max_its,
      float const p0,
      float const p1,
      X &img,
      Log &log)
  {
    log.info(FMT_STRING("Starting Variably Preconditioned Conjugate Gradients"));
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    auto const dims = img.dimensions();
    X b(dims);
    X q(dims);
    X p(dims);
    X r(dims);
    X r1(dims);
    b.setZero();
    q.setZero();
    p = img;
    r = img;
    float const r0norm = Norm2(img);

    for (long icg = 0; icg < max_its; icg++) {
      float const prog = static_cast<float>(icg) / ((max_its == 1) ? 1. : (max_its - 1.f));
      float const pre = std::exp(std::log(pre1) * prog + std::log(pre0) * (1.f - prog));
      sys(p, q, pre);
      r1 = r;
      float const r_old = Norm2(r1);
      float const alpha = r_old / std::real(Dot(p, q));
      b.device(dev) = b + p * p.constant(alpha);

      r.device(dev) = r - q * q.constant(alpha);
      float const beta = std::real(Dot(r, r - r1)) / r_old;
      p.device(dev) = r + p * p.constant(beta);
      float const delta = Norm2(r) / r0norm;
      log.image(b, fmt::format(FMT_STRING("cg-b-{:02}.nii"), icg));
      log.info(FMT_STRING("CG {}: ɑ {} β {} δ {} pre {}"), icg, alpha, beta, delta, pre);
      if (delta < thresh) {
        log.info(FMT_STRING("Reached convergence threshold"));
        break;
      }
    }
    img = b;
  }
};