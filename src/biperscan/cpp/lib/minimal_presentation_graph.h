#ifndef BIPERSCAN_LIB_MINIMAL_PRESENTATION_GRAPH_H
#define BIPERSCAN_LIB_MINIMAL_PRESENTATION_GRAPH_H

#include <algorithm>

#include "bigrade_ordering.h"
#include "concepts.h"

namespace bppc::detail {

// Wrapper around minpres to access it as an undirected graph
template <
    templated_view_of<edge_t> edge_view_t,
    templated_view_of<bigrade_t> grade_view_t>
  requires std::ranges::random_access_range<edge_view_t> and
           std::ranges::random_access_range<grade_view_t>
class minpres_graph_t {
  // Type shorthand
  using index_t =
      typename template_type<std::ranges::range_value_t<edge_view_t>>::type;
  using grade_t =
      typename template_type<std::ranges::range_value_t<grade_view_t>>::type;
  using edge_t = edge_t<index_t>;
  using bigrade_t = bigrade_t<grade_t>;
  using value_t = std::pair<grade_t, index_t>;

  // Private data members
  grade_view_t const d_grades{};
  edge_view_t const d_edges{};
  std::vector<std::vector<index_t>> d_column_indices{};

 public:
  minpres_graph_t(
      grade_view_t const grades, edge_view_t const edges, std::size_t num_points
  )
      : d_grades(grades), d_edges(edges), d_column_indices(num_points) {
    for (auto const &[idx, edge] : std::views::enumerate(edges)) {
      d_column_indices[edge.parent].push_back(static_cast<index_t>(idx));
      d_column_indices[edge.child].push_back(static_cast<index_t>(idx));
    }
  }

  // Matrix like access
  [[nodiscard]] std::size_t num_points() const {
    return d_column_indices.size();
  }
  [[nodiscard]] std::size_t num_edges() const {
    return d_grades.size();
  }
  index_t column_limit(bigrade_t const &grade) const {
    return static_cast<index_t>(std::ranges::distance(
        d_grades.begin(),
        std::ranges::upper_bound(d_grades, grade, bigrade_lex_less<grade_t>{})
    ));
  }
  [[nodiscard]] std::pair<bigrade_t, edge_t> edge_at(index_t const idx) const {
    return std::make_pair(d_grades[idx], d_edges[idx]);
  }

  // Graph like access
  [[nodiscard]] std::vector<index_t> const &children(index_t const parent
  ) const {
    return d_column_indices[parent];
  }
  [[nodiscard]] std::pair<grade_t, index_t> child_at(
      index_t const parent, index_t const column
  ) const {
    edge_t const &edge = d_edges[column];
    return std::make_pair(
        d_grades[column].distance,
        edge.parent == parent ? edge.child : edge.parent
    );
  }
};

}  // namespace bppc::detail

#endif  // BIPERSCAN_LIB_MINIMAL_PRESENTATION_GRAPH_H
