#pragma once

#include "types.h"
#include "log.h"

namespace Sim {

// A simple struct for returning multiple things from a simulation without tuple
struct Result
{
  Eigen::MatrixXf dynamics;
  Eigen::MatrixXf parameters;
};

// Another simple struct for passing around the main ZTE sequence parameters
struct Sequence
{
  long sps;
  float alpha, TR, Tramp, Tssi, TI, Trec;
};

// Arg lists are getting annoyingly long
struct Parameter
{
  long N;
  float lo, hi;
  bool logspaced;

  float value(long const ii) const;

private:
  float linspace(long const i) const;
  float logspace(long const i) const;
};

template<int NP>
struct ParameterGenerator
{
  using Parameters = Eigen::Array<float, NP, 1>;

  ParameterGenerator(std::array<Parameter, NP> const &p) :
    pars{p}
  {}

  long totalN() const {
    long total = 1;
    for (auto const &par : pars) {
      total *= par.N;
    }
    return total;
  }

  Parameters values(long const ii) {
    assert(ii < totalN());
    Parameters p;
    long remaining = ii;
    for (long ip = 0; ip < NP; ip++) {
      long const parN = remaining % pars[ip].N;
      remaining /= pars[ip].N;
      p(ip) = pars[ip].value(parN);
    }
    return p;
  }

  Parameters rand() const {
    Parameters p;
    p.setRandom();
    for (long ip = 0; ip < NP; ip++) {
      p(ip) = pars[ip].lo + p(ip) * (pars[ip].hi - pars[ip].lo);
    }
    return p;
  }

private:
  std::array<Parameter, NP> pars;
};

}
