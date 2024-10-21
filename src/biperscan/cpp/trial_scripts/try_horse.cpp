#include <chrono>
#include <iostream>

#include "../tests/npy.h"
#include "biperscan.h"
#include "lib/ostream.h"

using namespace std::chrono;
using namespace std::ranges;

template <typename dist, typename lens>
void run_test(
    std::vector<dist> const &distances, std::vector<lens> const &lens_values
) {
  using namespace bppc;
  auto start = high_resolution_clock::now();
  biperscan_minpres_result_t res = biperscan_minpres<unsigned>(
      distances, lens_values
  );
  auto stop = high_resolution_clock::now();

  std::cout << "Num edges:  " << distances.size() << "\n";
  std::cout << "Num points: " << lens_values.size() << "\n";
  std::cout << "Time spent: "
            << duration_cast<milliseconds>(stop - start).count()
            << " milliseconds\n";

  start = high_resolution_clock::now();
  minimal_presentation_merges_t merges = biperscan_merges(
      res.minimal_presentation, lens_values.size(), 40, 0.25
  );
  stop = high_resolution_clock::now();
  std::cout << "\nTime spent: "
            << duration_cast<milliseconds>(stop - start).count()
            << " milliseconds\n";
  std::cout << "Num merges: " << merges.size() << "\n\n";

  {
    std::ofstream outfile("../../minpres.txt");
    print_table(outfile, res.minimal_presentation) << std::endl;
    outfile.close();
  }
}

int main() {
  std::vector<double> const distances =
      npy::load<double>("../../tests/data/horse_distance.npy");
  std::vector<double> const lens =
      npy::load<double>("../../tests/data/horse_lens.npy");
  run_test(distances, lens);

  std::vector<float> const dists_float{distances.begin(), distances.end()};
  std::vector<float> const lens_float{lens.begin(), lens.end()};
  run_test(dists_float, lens_float);
}
