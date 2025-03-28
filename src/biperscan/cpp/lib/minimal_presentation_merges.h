#ifndef BIPERSCAN_LIB_MINIMAL_PRESENTATION_MERGES_H
#define BIPERSCAN_LIB_MINIMAL_PRESENTATION_MERGES_H

#include "minimal_presentation_iteration_state.h"

namespace bppc {

template <std::unsigned_integral index_t, std::unsigned_integral grade_t>
class minimal_presentation_merges_t {
  // Type shorthands
  using edge_t = edge_t<index_t>;
  using merge_t = merge_t<index_t>;
  using bigrade_t = bigrade_t<grade_t>;
  using value_t = std::pair<bigrade_t, edge_t>;

  // Private data members
  std::vector<index_t> d_start_columns{};
  std::vector<index_t> d_end_columns{};
  std::vector<grade_t> d_lens_grades{};
  std::vector<grade_t> d_distance_grades{};
  std::vector<index_t> d_parents{};
  std::vector<index_t> d_children{};
  std::vector<std::vector<index_t>> d_parent_sides{};
  std::vector<std::vector<index_t>> d_child_sides{};

 public:
  template <view_of<edge_t> edge_view_t, view_of<bigrade_t> grade_view_t>
  minimal_presentation_merges_t(
      grade_view_t const grades, edge_view_t const edges,
      std::size_t const num_points, std::size_t const min_cluster_size = 5,
      double const limit_fraction = 1.0
  ) {
    detail::minpres_graph_t graph{grades, edges, num_points};
    grade_t const upper_distance_limit = max_distance_grade(
        limit_fraction, num_points
    );

    // Find merges in the minimal presentation
    // reserve(grades.size()); // TODO: what is a reasonable size?
    std::vector<std::optional<index_t>> prev_edges(num_points);
    for (auto const &[idx, edge] : std::views::enumerate(edges)) {
      std::optional<index_t> &prev_edge = prev_edges[edge.child];
      if (prev_edge)
        check_for_merges(
            graph, *prev_edge, static_cast<index_t>(idx), min_cluster_size,
            upper_distance_limit
        );
      prev_edge = static_cast<index_t>(idx);
    }
    shrink_to_fit();
  }

  // View of merges in detection order (bigrade_lex_less within each merge)!
  [[nodiscard]] view_of<std::pair<bigrade_t, merge_t>> auto items() const {
    return std::views::zip_transform(
        std::make_pair<bigrade_t, merge_t>, grades(), merges()
    );
  }
  [[nodiscard]] view_of<bigrade_t> auto grades() const {
    return std::views::zip_transform(
        [](auto... args) { return bigrade_t{args...}; },
        std::views::all(d_lens_grades), std::views::all(d_distance_grades)
    );
  }
  [[nodiscard]] view_of<edge_t> auto edges() const {
    return std::views::zip_transform(
        [](auto... args) { return edge_t{args...}; },
        std::views::all(d_parents), std::views::all(d_children)
    );
  }
  [[nodiscard]] view_of<merge_t> auto merges() const {
    return std::views::zip_transform(
        [](index_t start_column, index_t end_column, index_t parent,
           index_t child, std::vector<index_t> const &parent_side,
           std::vector<index_t> const &child_side) {
          return merge_t{
              start_column,
              end_column,
              parent,
              child,
              std::span<index_t const>{parent_side.begin(), parent_side.end()},
              std::span<index_t const>{child_side.begin(), child_side.end()}
          };
        },
        std::views::all(d_start_columns), std::views::all(d_end_columns),
        std::views::all(d_parents), std::views::all(d_children),
        std::views::all(d_parent_sides), std::views::all(d_child_sides)
    );
  }

  // Memory management
  [[nodiscard]] bool empty() const {
    return d_parents.empty();
  }
  [[nodiscard]] std::size_t size() const {
    return d_parents.size();
  }
  void clear() {
    d_lens_grades.clear();
    d_distance_grades.clear();
    d_parents.clear();
    d_children.clear();
    d_parent_sides.clear();
    d_child_sides.clear();
  }

  // Python API takes ownership of the vectors to avoid copies.
  [[nodiscard]] std::vector<index_t> &&take_start_columns() {
    return std::move(d_start_columns);
  }
  [[nodiscard]] std::vector<index_t> &&take_end_columns() {
    return std::move(d_end_columns);
  }
  [[nodiscard]] std::vector<grade_t> &&take_lens_grades() {
    return std::move(d_lens_grades);
  }
  [[nodiscard]] std::vector<grade_t> &&take_distance_grades() {
    return std::move(d_distance_grades);
  }
  [[nodiscard]] std::vector<index_t> &&take_parents() {
    return std::move(d_parents);
  }
  [[nodiscard]] std::vector<index_t> &&take_children() {
    return std::move(d_children);
  }
  [[nodiscard]] std::vector<std::vector<index_t>> &&take_parent_sides() {
    return std::move(d_parent_sides);
  }
  [[nodiscard]] std::vector<std::vector<index_t>> &&take_child_sides() {
    return std::move(d_child_sides);
  }

 private:
  template <view_of<edge_t> edge_view_t, view_of<bigrade_t> grade_view_t>
  void add_merge(
      std::pair<index_t, index_t> const &columns, bigrade_t const &grade,
      detail::minpres_iterator_t<edge_view_t, grade_view_t> &side_one,
      detail::minpres_iterator_t<edge_view_t, grade_view_t> &side_two
  ) {
    d_lens_grades.push_back(grade.lens);
    d_distance_grades.push_back(grade.distance);

    auto [parent_side, child_side] = side_one.root() <= side_two.root()
                                         ? std::tie(side_one, side_two)
                                         : std::tie(side_two, side_one);

    d_start_columns.push_back(columns.first);
    d_end_columns.push_back(columns.second);
    d_parents.push_back(parent_side.root());
    d_children.push_back(child_side.root());
    d_parent_sides.emplace_back(parent_side.take_children());
    d_child_sides.emplace_back(child_side.take_children());
    std::ranges::sort(d_parent_sides.back());
    std::ranges::sort(d_child_sides.back());
  }
  void reserve(std::size_t const size) {
    d_lens_grades.reserve(size);
    d_distance_grades.reserve(size);
    d_parents.reserve(size);
    d_children.reserve(size);
    d_parent_sides.reserve(size);
    d_child_sides.reserve(size);
  }
  void shrink_to_fit() {
    d_lens_grades.shrink_to_fit();
    d_distance_grades.shrink_to_fit();
    d_parents.shrink_to_fit();
    d_children.shrink_to_fit();
    d_parent_sides.shrink_to_fit();
    d_child_sides.shrink_to_fit();
  }

  [[nodiscard]] static grade_t max_distance_grade(
      double const limit_fraction, std::size_t const num_points
  ) {
    return static_cast<grade_t>(
        limit_fraction *
        static_cast<double>(num_points * num_points - num_points) / 2
    );
  }
  template <view_of<edge_t> edges_view_t, view_of<bigrade_t> grades_view_t>
  void check_for_merges(
      detail::minpres_graph_t<edges_view_t, grades_view_t> const &graph,
      index_t const idx_one, index_t const idx_two,
      std::size_t const min_cluster_size, grade_t const upper_distance_limit
  ) {
    // Extract grades and edges
    auto [grade_one, edge_one] = graph.edge_at(idx_one);
    auto [grade_two, edge_two] = graph.edge_at(idx_two);

    if (edge_one.parent == edge_two.parent) {
      traverse_graph(
          graph, std::make_pair(idx_one, idx_two), grade_two, edge_one,
          std::optional<index_t>{}, min_cluster_size, upper_distance_limit
      );
    } else {
      traverse_graph(
          graph, std::make_pair(idx_one, idx_two),
          bigrade_t{grade_two.lens, grade_one.distance},
          edge_t{edge_one.parent, edge_two.parent},
          std::make_optional(edge_one.child), min_cluster_size,
          upper_distance_limit
      );
    }
  }
  template <view_of<edge_t> edges_view_t, view_of<bigrade_t> grades_view_t>
  void traverse_graph(
      detail::minpres_graph_t<edges_view_t, grades_view_t> const &graph,
      std::pair<index_t, index_t> const &columns, bigrade_t grade,
      edge_t const &roots, std::optional<index_t> const connecting_point,
      std::size_t const min_cluster_size, grade_t const upper_distance_limit
  ) {
    // Skip potential merge if distance is too large
    if (grade.distance >= upper_distance_limit)
      return;

    // Initialize iteration states
    index_t const column_limit = graph.column_limit(grade);
    detail::minpres_iterator_t<edges_view_t, grades_view_t> side_one{
        graph, roots.parent, connecting_point, column_limit, grade.distance
    };
    detail::minpres_iterator_t<edges_view_t, grades_view_t> side_two{
        graph, roots.child, connecting_point, column_limit, grade.distance
    };

    // Collect children on both sides until they converge
    while ((not side_one.empty() or not side_two.empty()) and
           (not side_one.empty() or side_one.num_children() >= min_cluster_size) and
           (not side_two.empty() or side_two.num_children() >= min_cluster_size)) {
      auto [active_side, other_side] = order_sides(side_one, side_two);
      if (grade_t const distance_bound = active_side.next_distance();
          active_side.collect_children(other_side, distance_bound)) {
        grade.distance = distance_bound;
        break;
      }
    }

    // Store merge if both sides have sufficient size
    if (side_one.num_children() >= min_cluster_size and
        side_two.num_children() >= min_cluster_size)
      add_merge(columns, grade, side_one, side_two);
  }
  template <typename iteration_type_t>
  [[nodiscard]] static std::pair<iteration_type_t &, iteration_type_t &>
  order_sides(iteration_type_t &side_one, iteration_type_t &side_two) {
    if (side_one.empty())
      return {side_two, side_one};
    if (side_two.empty())
      return {side_one, side_two};
    if (side_one.next_distance() <= side_two.next_distance())
      return {side_one, side_two};
    return {side_two, side_one};
  }
};

// Deduction guides
template <
    std::ranges::sized_range grade_range_t,
    std::ranges::sized_range edge_range_t>
minimal_presentation_merges_t(
    grade_range_t, edge_range_t, std::size_t, std::size_t, double
)
    -> minimal_presentation_merges_t<
        typename detail::template_type<
            std::ranges::range_value_t<edge_range_t>>::type,
        typename detail::template_type<
            std::ranges::range_value_t<grade_range_t>>::type>;

}  // namespace bppc

#endif  // BIPERSCAN_LIB_MINIMAL_PRESENTATION_MERGES_H
