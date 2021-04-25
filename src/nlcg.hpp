#pragma once

#include "log.h"
#include "tensorOps.h"
#include "types.h"

// Nonlinear Conjugate Gradient
template <int XDim, int YDim>
struct NLCG
{
  using X = Eigen::Tensor<Cx, XDim>;
  using Y = Eigen::Tensor<Cx, YDim>;
  using Residual = std::function<float(X const &x, Y &y)>;
  using JacAdjoint = std::function<void(X const &x, Y const &y, X &j)>;

  static void
  Run(Residual const &f,
      JacAdjoint const &jac,
      long const &max_its,
      float const &thresh,
      typename Y::Dimensions const &ydims,
      X &x,
      Log &log)
  {
    auto dev = Threads::GlobalDevice();

    Y residual(ydims);
    // Calculate cost f and dx at same time
    auto f_dx = [&](X const &x, X &dx) -> float {
      float const r2 = f(x, residual);
      jac(x, residual, dx);
      return r2;
    };

    auto linesearch =
        [&](X const &xx, X const &pp, float const a0, float const m, float const fx) -> float {
      // Line search
      float const c = 0.5f;   // Line-search threshold
      float const rho = 0.5f; // Line-search step size will be reduced by this each iteration
      float const t = c * m;
      float alpha = a0;
      X xs(xx.dimensions()); // Line-search position
      log.info(FMT_STRING("Start linesearch. Slope {} residual {}"), m, fx);
      float fxs = 0;
      float thresh = 0;
      for (long ii = 0; ii < 10; ii++) {
        xs.device(dev) = xx + xx.constant(alpha) * pp;
        fxs = f(xs, residual);
        thresh = fx + alpha * t;
        log.info(FMT_STRING("ɑ {} residual {} thresh {}"), alpha, fxs, thresh);
        if (fxs < thresh) {
          break;
        } else {
          alpha *= rho;
        }
      }
      return alpha;
    };

    auto const dims = x.dimensions();
    X dx(dims);
    X dx1(dims);
    X p(dims);
    dx.setZero();
    dx1.setZero();
    p.setZero();
    // Initial search direction
    float r2 = f_dx(x, dx);
    p = -dx;
    log.image(x, "nlcg-x-00.nii");
    log.image(p, "nlcg-p-00.nii");
    float const r0 = r2;
    float a0 = 1.f;
    log.info(FMT_STRING("NLCG init residual {}"), r0);
    for (long ii = 1; ii < max_its; ii++) {
      float const m = Dot(dx, p).real();
      float const alpha = linesearch(x, p, a0, m, r2);
      x.device(dev) = x + alpha * p;
      // Polak - Ribierre update
      dx1 = dx;
      r2 = f_dx(x, dx);
      float const beta = 0.f; // std::max(0.f, std::real(Dot(dx, dx - dx1)) / Norm2(dx1));
      p.device(dev) = -dx + beta * p;
      log.info("Iteration {:02d} β {} residual {} threshold {}", ii, beta, r2, r0 * thresh);
      a0 = alpha * m / Dot(dx, p).real();
      log.image(x, fmt::format(FMT_STRING("nlcg-x-{:02d}.nii"), ii));
      log.image(p, fmt::format(FMT_STRING("nlcg-p-{:02d}.nii"), ii));
      log.image(dx, fmt::format(FMT_STRING("nlcg-dx-{:02d}.nii"), ii));
      if ((r2 / r0) < thresh) {
        log.info("Iteration threshold reached");
        break;
      }
    }
  }
};
