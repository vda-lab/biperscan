#include <array>
#include <iostream>

#include "biperscan.h"
#include "lib/ostream.h"

int main() {
  using namespace bppc;
  constexpr std::array dists{1.5,        1.80277564, 1.,         0.90138782,
                             1.,         1.80277564, 0.90138782, 1.5,
                             0.90138782, 0.90138782};
  constexpr std::array lens{1., 1., 1., 1., 2.};

  biperscan_minpres_result_t<unsigned, std::size_t> res =
      biperscan_minpres<unsigned, std::size_t>(dists, lens);
  print_table(std::cout, res.minimal_presentation) << std::endl;

  minimal_presentation_merges_t merges = biperscan_merges(
      res.minimal_presentation, lens.size(), 2, 1.0
  );
  std::cout << "Merges:\n";
  for (auto const &[grade, merge] : merges.items())
    std::cout << grade << ' ' << merge.roots << '\n';

  // Not needed for the pipeline any more, just demonstrating how to call it.
  linkage_hierarchy_t hierarchy = biperscan_linkage(
      res.minimal_presentation, lens.size()
  );
  std::cout << "\nHierarchy:\n";
  std::cout << hierarchy << std::endl;
}
