#pragma once

#include <Eigen/Core>

Eigen::MatrixXcf LowRank(Eigen::Map<Eigen::MatrixXcf> const &m, long const nRetain);