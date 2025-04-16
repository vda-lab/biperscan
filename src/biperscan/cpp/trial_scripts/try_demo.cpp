#include <array>
#include <iostream>

#include "biperscan.h"
#include "lib/ostream.h"

int main() {
  using namespace bppc;
  constexpr std::array dists{4.,         1.,         3.49284984, 3.04795013,
                             2.05182845, 3.49284984, 1.,         2.62488095,
                             3.0016662,  2.8,        2.0808652,  1.06301458,
                             1.62788206, 2.11896201, 1.02956301};
  constexpr std::array lens{0., 0., 0.50799656, 0.50799656, 0.95363885, 1.};

  biperscan_minpres_result_t<unsigned, std::size_t> res =
      biperscan_minpres<unsigned, std::size_t>(dists, lens);
  print_table(std::cout, res.minimal_presentation) << std::endl;

  minimal_presentation_merges_t merges = biperscan_merges(
      res.minimal_presentation, lens.size(), 2, 1.0
  );
  std::cout << "Merges:\n";
  for (auto const &[grade, merge] : merges.items())
    std::cout << grade << ' ' << edge_t{merge.parent, merge.child} << '\n';

  // Not needed for the pipeline any more, just demonstrating how to call it.
  linkage_hierarchy_t hierarchy = biperscan_linkage(
      res.minimal_presentation, lens.size()
  );
  std::cout << "\nHierarchy:\n";
  std::cout << hierarchy << std::endl;
}
