#include <array>
#include <iostream>
#include <ranges>

#include "lib/ostream.h"
#include "py_biperscan.h"

using namespace bppc;

int main() {
  constexpr std::array ranks{3., 4., 2., 1., 2., 4., 1., 3., 1., 1.};
  constexpr std::array lens{1., 1., 1., 1., 2.};

  auto result = python::biperscan_minpres<unsigned>(
      ranks.data(), ranks.size(), lens.data(), lens.size()
  );

  std::cout << "Minimal presentation\n";
  for (auto const &[lens, distance, parent, child] : std::views::zip(
           result.minpres_lens_grades, result.minpres_distance_grades,
           result.minpres_parents, result.minpres_children
       )) {
    std::cout << bigrade_t{lens, distance} << " " << edge_t{parent, child}
              << '\n';
  }

  auto m_result = python::biperscan_merges(
      result.minpres_lens_grades.data(), result.minpres_distance_grades.data(),
      result.minpres_parents.data(), result.minpres_children.data(),
      result.minpres_lens_grades.size(), lens.size(), 2, 1.0
  );
  std::cout << "\nMerges:\n";
  for (auto const &[lens, distance, parent, child] : std::views::zip(
           m_result.merge_lens_grades, m_result.merge_distance_grades,
           m_result.merge_parents, m_result.merge_children
       ))
    std::cout << bigrade_t{lens, distance} << ' ' << edge_t{parent, child}
              << '\n';

  // Not needed for the pipeline any more, just demonstrating how to call it.
  auto l_result = python::biperscan_linkage(
      result.minpres_lens_grades.data(), result.minpres_distance_grades.data(),
      result.minpres_parents.data(), result.minpres_children.data(),
      result.minpres_lens_grades.size(), lens.size()
  );
  std::cout << "\nLinkage hierarchy\n";
  std::size_t id = lens.size();
  for (auto const &[lens, distance, parent, child, parent_root, child_root] :
       std::views::zip(
           l_result.linkage_lens_grades, l_result.linkage_distance_grades,
           l_result.linkage_parents, l_result.linkage_children,
           l_result.linkage_parent_roots, l_result.linkage_child_roots
       )) {
    std::cout << bigrade_t{lens, distance} << ' '
              << link_t{id++, parent, child, parent_root, child_root} << '\n';
  }
}
