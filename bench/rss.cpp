#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

TEST_CASE("RSS")
{
  int const sz = 128;
  int const hsz = sz / 2;
  int const qsz = sz / 4;
  using Cx4 = Eigen::Tensor<std::complex<float>, 4>;
  using Cx3 = Eigen::Tensor<std::complex<float>, 3>;
  using Dims4 = Cx4::Dimensions;
  Cx4 grid(16, sz, sz, sz);
  grid.setRandom();
  Cx3 image(hsz, hsz, hsz);
  image.setZero();
  Eigen::ThreadPool gp(std::thread::hardware_concurrency());
  Eigen::ThreadPoolDevice dev(&gp, gp.NumThreads());

  BENCHMARK("Naive")
  {
    image =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .sum(Eigen::array<int, 1>{0})
            .sqrt();
  };

  BENCHMARK("Naive-MT")
  {
    image.device(dev) =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .sum(Eigen::array<int, 1>{0})
            .sqrt();
  };

  BENCHMARK("IndexList1")
  {
    image =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .sum(Eigen::IndexList<Eigen::type2index<0>>())
            .sqrt();
  };

  BENCHMARK("IndexList1-MT")
  {
    image.device(dev) =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .sum(Eigen::IndexList<Eigen::type2index<0>>())
            .sqrt();
  };

  BENCHMARK("IndexList2")
  {
    Eigen::IndexList<Eigen::type2index<0>, int, int, int> start;
    start.set(1, qsz);
    start.set(2, qsz);
    start.set(3, qsz);
    image = (grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
             grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
                .sum(Eigen::IndexList<Eigen::type2index<0>>())
                .sqrt();
  };

  BENCHMARK("IndexList2-MT")
  {
    Eigen::IndexList<Eigen::type2index<0>, int, int, int> start;
    start.set(1, qsz);
    start.set(2, qsz);
    start.set(3, qsz);
    image.device(dev) = (grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
                         grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
                            .sum(Eigen::IndexList<Eigen::type2index<0>>())
                            .sqrt();
  };
}