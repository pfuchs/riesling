#pragma once

namespace HD5 {

void Init();
hid_t InfoType();
void CheckInfoType(hid_t handle);

namespace Keys {
std::string const Info = "info";
std::string const Meta = "meta";
std::string const Noncartesian = "noncartesian";
std::string const Cartesian = "cartesian";
std::string const Image = "image";
std::string const Trajectory = "trajectory";
std::string const Basis = "basis";
std::string const BasisImages = "basis-images";
std::string const Dynamics = "dynamics";
std::string const SDC = "sdc";
std::string const SENSE = "sense";
} // namespace Keys

} // namespace HD5
