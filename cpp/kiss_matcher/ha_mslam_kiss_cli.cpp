#include <kiss_matcher/KISSMatcher.hpp>

#include <Eigen/Core>

#include <tbb/global_control.h>
#include <omp.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<Eigen::Vector3f> readMetricCloud(const std::string & path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed_to_open_cloud");
  }
  std::uint64_t count = 0U;
  stream.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (!stream || count < 3U) {
    throw std::runtime_error("invalid_cloud_header");
  }
  std::vector<Eigen::Vector3f> points;
  points.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0U; index < count; ++index) {
    double xyz[3]{};
    stream.read(reinterpret_cast<char *>(xyz), sizeof(xyz));
    if (!stream) {
      throw std::runtime_error("truncated_cloud");
    }
    const Eigen::Vector3f point(
      static_cast<float>(xyz[0]),
      static_cast<float>(xyz[1]),
      static_cast<float>(xyz[2]));
    if (point.allFinite()) {
      points.push_back(point);
    }
  }
  if (points.size() < 3U) {
    throw std::runtime_error("insufficient_finite_points");
  }
  return points;
}

void printTransform(
  const Eigen::Matrix3d & rotation,
  const Eigen::Vector3d & translation)
{
  std::cout << "[";
  for (int row = 0; row < 4; ++row) {
    if (row != 0) {
      std::cout << ",";
    }
    std::cout << "[";
    for (int column = 0; column < 4; ++column) {
      if (column != 0) {
        std::cout << ",";
      }
      double value = 0.0;
      if (row < 3 && column < 3) {
        value = rotation(row, column);
      } else if (row < 3 && column == 3) {
        value = translation(row);
      } else if (row == 3 && column == 3) {
        value = 1.0;
      }
      std::cout << value;
    }
    std::cout << "]";
  }
  std::cout << "]";
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 4) {
    std::cerr << "Usage: ha_mslam_kiss_cli VOXEL_SIZE SOURCE_BIN TARGET_BIN\n";
    return 2;
  }

  try {
    const float voxel_size = std::stof(argv[1]);
    if (!(voxel_size > 0.0F)) {
      throw std::runtime_error("invalid_voxel_size");
    }

    tbb::global_control control(
      tbb::global_control::max_allowed_parallelism, 1U);
    omp_set_dynamic(0);
    omp_set_num_threads(1);

    const auto source = readMetricCloud(argv[2]);
    const auto target = readMetricCloud(argv[3]);

    kiss_matcher::KISSMatcherConfig config(
      voxel_size,
      false,   // already voxelized by HA-MSLAM
      false,   // full SE(3) GNC-TLS, not QUATRO
      1.0F,    // disable linearity filtering
      5000,
      3.0F,
      5.0F,
      1.0F,
      0.75F,
      true);

    kiss_matcher::KISSMatcher matcher(config);
    const auto solution = matcher.estimate(source, target);

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"valid\":" << (solution.valid ? "true" : "false") << ",";
    std::cout << "\"source_points\":" << source.size() << ",";
    std::cout << "\"target_points\":" << target.size() << ",";
    std::cout << "\"initial_correspondences\":"
              << matcher.getInitialCorrespondences().size() << ",";
    std::cout << "\"final_correspondences\":"
              << matcher.getFinalCorrespondences().size() << ",";
    std::cout << "\"rotation_inliers\":"
              << matcher.getNumRotationInliers() << ",";
    std::cout << "\"final_inliers\":"
              << matcher.getNumFinalInliers() << ",";
    std::cout << "\"T_target_source\":";
    printTransform(solution.rotation, solution.translation);
    std::cout << "}\n";

    return solution.valid ? 0 : 5;
  } catch (const std::exception & error) {
    std::cerr << error.what() << "\n";
    return 6;
  }
}
