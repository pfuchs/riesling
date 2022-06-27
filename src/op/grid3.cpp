#include "grid.hpp"

std::unique_ptr<GridBase> make_3(Kernel const *k, Mapping const &m, Index const nC)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<Grid<3, 1>>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, nC);
  } else {
    return std::make_unique<Grid<3, 3>>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, nC);
  }
}
