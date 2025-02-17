/*
 * threads.cpp
 *
 * Copyright (c) 2019 Tobias Wood
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include "threads.hpp"
#include "log.hpp"

// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace {
Eigen::ThreadPool *gp = nullptr;
}

namespace rl {
namespace Threads {

Eigen::ThreadPool *GlobalPool()
{
  if (gp == nullptr) {
    auto const nt = std::thread::hardware_concurrency();
    Log::Print<Log::Level::High>("Creating default thread pool with {} threads", nt);
    gp = new Eigen::ThreadPool(nt);
  }
  return gp;
}

void SetGlobalThreadCount(Index nt)
{
  if (gp) { delete gp; }
  if (nt < 1) { nt = std::thread::hardware_concurrency(); }
  Log::Print<Log::Level::High>("Creating thread pool with {} threads", nt);
  gp = new Eigen::ThreadPool(nt);
}

Index GlobalThreadCount() { return GlobalPool()->NumThreads(); }

Eigen::ThreadPoolDevice GlobalDevice() { return Eigen::ThreadPoolDevice(GlobalPool(), GlobalPool()->NumThreads()); }

void For(ForFunc f, Index const lo, Index const hi, std::string const &label)
{
  Index const ni = hi - lo;
  Index const nt = GlobalPool()->NumThreads();
  if (ni == 0) { return; }

  bool const report = label.size();
  if (report) { Log::StartProgress(ni, label); }
  if (nt == 1) {
    for (Index ii = lo; ii < hi; ii++) {
      f(ii);
      if (report) { Log::Tick(); }
    }
  } else {
    Eigen::Barrier barrier(static_cast<unsigned int>(ni));
    for (Index ii = lo; ii < hi; ii++) {
      GlobalPool()->Schedule([&barrier, &f, ii, report] {
        f(ii);
        barrier.Notify();
        if (report) { Log::Tick(); }
      });
    }
    barrier.Wait();
  }
  if (report) { Log::StopProgress(); }
}

void For(ForFunc f, Index const n, std::string const &label) { For(f, 0, n, label); }

} // namespace Threads
} // namespace rl
