#ifndef BIPERSCAN_LIB_MINIMAL_PRESENTATION_H
#define BIPERSCAN_LIB_MINIMAL_PRESENTATION_H

#include <algorithm>
#include <optional>

#include "graded_matrix.h"

namespace bppc {

/**
 * @brief A minimal presentation constructed from graded edges.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 */
template <std::unsigned_integral index_t, std::unsigned_integral grade_t>
class minimal_presentation_t {
  // Type shorthands
  using edge_t = edge_t<index_t>;
  using bigrade_t = bigrade_t<grade_t>;
  using value_t = std::pair<bigrade_t, edge_t>;
  using graded_matrix_t = graded_matrix_t<index_t, grade_t>;
  using pivot_t = typename graded_matrix_t::iterator_t;

  // Private data members.
  std::vector<grade_t> d_lens_grades{};
  std::vector<grade_t> d_distance_grades{};
  std::vector<index_t> d_parents{};
  std::vector<index_t> d_children{};

 public:
  // Construct from known edges (with sanity checks).
  template <range_of<value_t> edge_range_t>
  minimal_presentation_t(std::from_range_t, edge_range_t &&edges) {
    graded_matrix_t matrix{std::from_range, std::forward<edge_range_t>(edges)};
    reserve(matrix.size());
    auto items_view = matrix.items();
    pivot_t column_it = items_view.begin();
    while (column_it != items_view.end())
      add_edge(column_it++);
  }

  // Construct from a graded matrix.
  minimal_presentation_t(graded_matrix_t matrix, std::size_t const num_points) {
    std::vector<std::optional<pivot_t>> pivots(num_points);

    // Graded matrix provides the edges in bigrade_lex_less order!
    reserve(matrix.size());
    auto items_view = matrix.items();
    for (pivot_t column_it = items_view.begin(); column_it != items_view.end();
         ++column_it) {
      if (reduce_column(pivots, column_it))
        add_edge(column_it);
    }
    shrink_to_fit();
  }

  [[nodiscard]] bool operator==(minimal_presentation_t const &) const = default;

  // Individual edge access.
  [[nodiscard]] edge_t edge_at(bigrade_t const &grade) const {
    auto view = grades();
    auto const it = std::ranges::lower_bound(
        view, grade, bigrade_lex_less<grade_t>{}
    );
    auto const idx = std::ranges::distance(view.begin(), it);
    return edge_t{d_parents[idx], d_children[idx]};
  }

  // Iteration views.
  [[nodiscard]] view_of<value_t> auto items() const {
    return std::views::zip_transform(
        std::make_pair<bigrade_t, edge_t>, grades(), edges()
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

  // Memory management.
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
  }

  // Python API takes ownership of the vectors to avoid copies.
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

 private:
  void add_edge(pivot_t const it) {
    d_lens_grades.push_back(it->first.lens);
    d_distance_grades.push_back(it->first.distance);
    d_parents.push_back(it->second.parent);
    d_children.push_back(it->second.child);
  }
  void reserve(std::size_t const size) {
    d_lens_grades.reserve(size);
    d_distance_grades.reserve(size);
    d_parents.reserve(size);
    d_children.reserve(size);
  }
  void shrink_to_fit() {
    d_lens_grades.shrink_to_fit();
    d_distance_grades.shrink_to_fit();
    d_parents.shrink_to_fit();
    d_children.shrink_to_fit();
  }

  [[nodiscard]] static bool reduce_column(
      std::vector<std::optional<pivot_t>> &pivots, pivot_t column_it
  ) {
    auto &[grade, edge] = *column_it;
    while (true) {
      std::optional<pivot_t> &pivot = pivots[edge.child];
      if (not pivot or (*pivot)->first.distance > grade.distance) {
        pivot = column_it;
        break;
      }

      index_t const pivot_parent = (*pivot)->second.parent;
      if (edge.parent == pivot_parent)
        return false;
      auto [parent, child] = std::minmax(edge.parent, pivot_parent);
      edge = edge_t{parent, child};
    };
    return true;
  }
};

template <detail::graded_range edge_range_t>
minimal_presentation_t(std::from_range_t, edge_range_t &&)
    -> minimal_presentation_t<
        typename detail::template_type<std::ranges::range_value_t<
            decltype(std::views::values(std::declval<edge_range_t>()))>>::type,
        typename detail::template_type<std::ranges::range_value_t<
            decltype(std::views::keys(std::declval<edge_range_t>()))>>::type>;

}  // namespace bppc

#endif  // BIPERSCAN_LIB_MINIMAL_PRESENTATION_H
