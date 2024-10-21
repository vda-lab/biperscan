#ifndef BIPERSCAN_LIB_GRADED_MATRIX_H
#define BIPERSCAN_LIB_GRADED_MATRIX_H

#include <map>
#include <set>
#include <stdexcept>

#include "bigrade_ordering.h"
#include "concepts.h"

namespace bppc {

/**
 * @brief A graded matrix constructed from distance & lens ranks ordered in
 * `bigrade_lex_less` for minimal presentation processing!
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 */
template <std::unsigned_integral index_t, std::unsigned_integral grade_t>
class graded_matrix_t {
  // Type shorthands
  using edge_t = edge_t<index_t>;
  using bigrade_t = bigrade_t<grade_t>;
  using value_t = std::pair<bigrade_t const, edge_t>;
  using matrix_t = std::map<bigrade_t, edge_t, bigrade_lex_less<grade_t>>;

  // Private data members.
  matrix_t d_edges{};

 public:
  using iterator_t = typename matrix_t::iterator;

  // Construct from known edges (with sanity checks).
  template <range_of<value_t> edge_range_t>
  graded_matrix_t(std::from_range_t, edge_range_t &&edges)
      : d_edges(std::ranges::to<matrix_t>(std::forward<edge_range_t>(edges))) {
    for (auto const &edge : this->edges())
      if (edge.child < edge.parent)
        throw std::runtime_error("Edge parent must be less than child!");

    std::set<grade_t> distances{};
    for (auto const &grade : this->grades())
      if (not distances.insert(grade.distance).second)
        throw std::runtime_error("Distance grades must be unique!");
  }

  // Construct from grades.
  template <
      range_of<grade_t> lens_range_t, range_of<grade_t> distance_range_t,
      range_of<index_t> index_range_t>
    requires std::ranges::random_access_range<lens_range_t> and
             std::ranges::random_access_range<distance_range_t> and
             std::ranges::random_access_range<index_range_t>
  graded_matrix_t(
      lens_range_t &&lens_grades, distance_range_t &&distance_grades,
      index_range_t &&point_to_row
  ) {
    for (auto const [grade, edge] : graded_edges(
             std::views::all(lens_grades), std::views::all(distance_grades),
             std::views::all(point_to_row)
         ))
      d_edges.emplace(grade, edge);
  }

  [[nodiscard]] bool operator==(graded_matrix_t const &) const = default;

  // Iteration views.
  [[nodiscard]] view_of<value_t> auto items() {
    return std::views::all(d_edges);
  }
  [[nodiscard]] view_of<value_t> auto items() const {
    return std::views::all(d_edges);
  }
  [[nodiscard]] view_of<bigrade_t> auto grades() const {
    return std::views::keys(d_edges);
  }
  [[nodiscard]] view_of<edge_t> auto edges() {
    return std::views::values(d_edges);
  }
  [[nodiscard]] view_of<edge_t> auto edges() const {
    return std::views::values(d_edges);
  }

  // Memory management
  [[nodiscard]] std::size_t size() const {
    return d_edges.size();
  }
  [[nodiscard]] bool empty() const {
    return d_edges.empty();
  }
  void clear() {
    return d_edges.clear();
  }

 private:
  template <
      view_of<grade_t> lens_range_t, view_of<grade_t> distance_range_t,
      view_of<index_t> index_range_t>
  [[nodiscard]] static auto graded_edges(
      lens_range_t lens_grades, distance_range_t distance_grades,
      index_range_t point_to_row
  ) {
    return std::views::zip_transform(
        [](grade_t const dist, auto const &&edge_item) {
          return std::pair<bigrade_t const, edge_t>{
              bigrade_t{edge_item.first, dist}, edge_item.second
          };
        },
        distance_grades, edges_from_grades(lens_grades, point_to_row)
    );
  }

  template <view_of<grade_t> lens_view_t, view_of<index_t> index_view_t>
  [[nodiscard]] static auto edges_from_grades(
      lens_view_t lens_grades, index_view_t row_to_point
  ) {
    return std::views::join(std::views::transform(
        std::views::iota(index_t{0}, std::ranges::size(lens_grades)),
        [lens_grades, row_to_point](index_t const row) {
          return std::views::transform(
              std::views::iota(
                  index_t{row} + 1, std::ranges::size(lens_grades)
              ),
              [row, &lens_grades, &row_to_point](index_t const col) {
                grade_t const lens = std::max(
                    lens_grades[row], lens_grades[col]
                );
                auto const [parent, child] = std::minmax(
                    row_to_point[row], row_to_point[col]
                );
                return std::make_pair(lens, edge_t{parent, child});
              }
          );
        }
    ));
  }
};

// Deduction guides.
template <detail::graded_range edge_range_t>
graded_matrix_t(std::from_range_t, edge_range_t &&) -> graded_matrix_t<
    typename detail::template_type<std::ranges::range_value_t<
        decltype(std::views::values(std::declval<edge_range_t>()))>>::type,
    typename detail::template_type<std::ranges::range_value_t<
        decltype(std::views::keys(std::declval<edge_range_t>()))>>::type>;

template <
    std::ranges::random_access_range lens_range_t,
    std::ranges::random_access_range distance_range_t,
    std::ranges::random_access_range index_range_t>
  requires std::same_as<
      std::ranges::range_value_t<lens_range_t>,
      std::ranges::range_value_t<distance_range_t>>
graded_matrix_t(
    lens_range_t &&lens_grades, distance_range_t &&distance_grades,
    index_range_t &&point_to_row
)
    -> graded_matrix_t<
        std::ranges::range_value_t<index_range_t>,
        std::ranges::range_value_t<lens_range_t>>;

}  // namespace bppc

#endif  // BIPERSCAN_LIB_GRADED_MATRIX_H
