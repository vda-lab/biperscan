#include <chrono>
#include <iostream>

#include "../tests/npy.h"
#include "biperscan.h"
#include "lib/minimal_presentation_merges.h"
#include "lib/ostream.h"

using namespace std::chrono;
using namespace std::ranges;
using namespace bppc;

int main() {
  std::vector<double> const distances =
      npy::load<double>("../../tests/data/flareable_new_dists.npy");
  std::vector<float> const lens_values =
      npy::load<float>("../../tests/data/flareable_new_lens.npy");

  auto start = high_resolution_clock::now();
  biperscan_minpres_result_t const res = biperscan_minpres<std::size_t>(
      distances, lens_values
  );
  auto stop = high_resolution_clock::now();
  std::size_t const num_points = lens_values.size();
  std::size_t const num_edges = res.minimal_presentation.size();
  std::cout << "Num edges:  " << distances.size() << "\n";
  std::cout << "Num points: " << num_points << "\n";
  std::cout << "Num minpres: " << num_edges << "\n";
  std::cout << "Time spent: "
            << duration_cast<milliseconds>(stop - start).count()
            << " milliseconds\n";

  start = high_resolution_clock::now();
  minimal_presentation_merges_t merges = biperscan_merges(
      res.minimal_presentation, lens_values.size(), 80, 0.04
  );
  stop = high_resolution_clock::now();
  std::cout << "\nTime spent: "
            << duration_cast<milliseconds>(stop - start).count()
            << " milliseconds\n";
  std::cout << "Num merges: " << merges.size() << "\n";
}
