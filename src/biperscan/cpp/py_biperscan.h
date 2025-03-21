#ifndef BIPERSCAN_PYTHON_API_H
#define BIPERSCAN_PYTHON_API_H

#include <chrono>
#include <span>

#include "biperscan.h"

namespace bppc::python {

/**
 * @brief Return type of `bppc::python::biperscan_minpres` containing the
 * minimal presentation separated into features vectors.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 */
template <std::unsigned_integral index_t, std::unsigned_integral grade_t>
struct biperscan_minpres_result_t {
  // Grades and mappings
  std::vector<index_t> col_to_edge{};
  std::vector<index_t> row_to_point{};
  std::vector<grade_t> lens_grades{};

  // Minpres
  std::vector<grade_t> minpres_lens_grades{};
  std::vector<grade_t> minpres_distance_grades{};
  std::vector<index_t> minpres_parents{};
  std::vector<index_t> minpres_children{};

  // Construction time information
  double matrix_time = 0.0;
  double minpres_time = 0.0;
};

/**
 * @brief Python / Cython wrapper for `biperscan_minpres`.
 * Steps:
 * - Compute bigraded matrix
 * - Compute bigraded minimal presentation
 * - Transforms output to format Cython understands
 * @param dist_ptr - A pointer a condensed distance matrix.
 * @param num_edges - The number of edges / length of the distance array.
 * @param lens_ptr - A pointer to a the data point lens values.
 * @param num_points - The number of points / length of the lens array.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @tparam dist_t  - Precision for storing distances.
 * @tparam lens_t - Precision for storing lens values.
 * @return A biperscan_minpres_result_t describing minimal presentation.
 */
template <
    std::unsigned_integral index_t, std::unsigned_integral grade_t = index_t,
    std::floating_point dist_t, std::floating_point lens_t>
biperscan_minpres_result_t<index_t, grade_t> biperscan_minpres(
    dist_t const *dist_ptr, std::size_t num_edges, lens_t const *lens_ptr,
    std::size_t num_points
) {
  // Create input ranges
  std::span distances{dist_ptr, dist_ptr + num_edges};
  std::span point_lens{lens_ptr, lens_ptr + num_points};

  // Construct graded matrix
  auto const matrix_start{std::chrono::steady_clock::now()};
  auto col_to_edge(argsort_of<index_t>(distances));
  auto row_to_point(argsort_of<index_t>(point_lens));
  auto lens_grades(dense_rank_from_argsort<grade_t>(point_lens, row_to_point));
  graded_matrix_t<index_t, grade_t> matrix{
      lens_grades, ordinal_rank_from_argsort<grade_t>(col_to_edge),
      ordinal_rank_from_argsort<index_t>(row_to_point)
  };
  auto const matrix_finish{std::chrono::steady_clock::now()};
  double const matrix_time =
      std::chrono::duration<double>(matrix_finish - matrix_start).count();

  // Construct minimal presentation
  auto const minpres_start{std::chrono::steady_clock::now()};
  minimal_presentation_t minpres{std::move(matrix), lens_grades.size()};
  auto const minpres_finish{std::chrono::steady_clock::now()};
  double const minpres_time =
      std::chrono::duration<double>(minpres_finish - minpres_start).count();

  return biperscan_minpres_result_t{
      std::move(col_to_edge),
      std::move(row_to_point),
      std::move(lens_grades),
      minpres.take_lens_grades(),
      minpres.take_distance_grades(),
      minpres.take_parents(),
      minpres.take_children(),
      matrix_time,
      minpres_time
  };
}

/**
 * @brief Return type of `bppc::python::biperscan_merges` containing the
 * detected merges separated into features vectors.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 */
template <std::unsigned_integral index_t, std::unsigned_integral grade_t>
struct biperscan_merge_result_t {
  std::vector<index_t> merge_start_columns{};
  std::vector<index_t> merge_end_columns{};
  std::vector<grade_t> merge_lens_grades{};
  std::vector<grade_t> merge_distance_grades{};
  std::vector<index_t> merge_parents{};
  std::vector<index_t> merge_children{};
  std::vector<std::vector<index_t>> merge_parent_sides{};
  std::vector<std::vector<index_t>> merge_child_sides{};

  // Construction time information
  double merge_time = 0.0;
};

/**
 * @brief Python / Cython wrapper for `biperscan_linkage`.
 * Steps:
 * - Compute bigraded matrix
 * - Compute bigraded minimal presentation
 * - Compute bigraded linkage hierarchy
 * - Transforms output to format Cython understands
 * @param lens_ptr - A pointer to the minpres lens grades.
 * @param dist_ptr - A pointer to the minpres distance grades.
 * @param parent_ptr - A pointer to the minpres parents.
 * @param child_ptr - A pointer to the minpres children.
 * @param num_edges - The number of edges in the minpres.
 * @param num_points - The number of points.
 * @param min_cluster_size - The minimum cluster size.
 * @param limit_fraction - The maximum distance grade fraction to use as
 * upper distance threshold.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @return A biperscan_linkage_result_t describing the linkage hierarchy.
 */
template <
    std::unsigned_integral index_t, std::unsigned_integral grade_t = index_t>
biperscan_merge_result_t<index_t, grade_t> biperscan_merges(
    grade_t const *lens_ptr, grade_t const *dist_ptr, index_t const *parent_ptr,
    index_t const *child_ptr, std::size_t const num_edges,
    std::size_t const num_points, std::size_t const min_cluster_size = 10,
    double const limit_fraction = 1.0
) {
  auto const start{std::chrono::steady_clock::now()};
  minimal_presentation_merges_t merges{
      std::views::zip_transform(
          [](auto &&...args) { return bigrade_t{args...}; },
          std::span{lens_ptr, lens_ptr + num_edges},
          std::span{dist_ptr, dist_ptr + num_edges}
      ),
      std::views::zip_transform(
          [](auto &&...args) { return edge_t{args...}; },
          std::span{parent_ptr, parent_ptr + num_edges},
          std::span{child_ptr, child_ptr + num_edges}
      ),
      num_points, min_cluster_size, limit_fraction
  };
  auto const finish{std::chrono::steady_clock::now()};

  return biperscan_merge_result_t{
      merges.take_start_columns(),
      merges.take_end_columns(),
      merges.take_lens_grades(),
      merges.take_distance_grades(),
      merges.take_parents(),
      merges.take_children(),
      merges.take_parent_sides(),
      merges.take_child_sides(),
      std::chrono::duration<double>(finish - start).count()  // seconds
  };
}

/**
 * @brief Return type of `bppc::python::biperscan_linkage` containing the
 * linkage hierarchy separated into features vectors.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 */
template <std::unsigned_integral index_t, std::unsigned_integral grade_t>
struct biperscan_linkage_result_t {
  std::vector<grade_t> linkage_lens_grades{};
  std::vector<grade_t> linkage_distance_grades{};
  std::vector<index_t> linkage_parents{};
  std::vector<index_t> linkage_children{};
  std::vector<index_t> linkage_parent_roots{};
  std::vector<index_t> linkage_child_roots{};

  // Construction time information
  double linkage_time = 0.0;
};

/**
 * @brief Python / Cython wrapper for `biperscan_linkage`.
 * Steps:
 * - Compute bigraded matrix
 * - Compute bigraded minimal presentation
 * - Compute bigraded linkage hierarchy
 * - Transforms output to format Cython understands
 * @param lens_ptr - A pointer to the minpres lens grades.
 * @param dist_ptr - A pointer to the minpres distance grades.
 * @param parent_ptr - A pointer to the minpres parents.
 * @param child_ptr - A pointer to the minpres children.
 * @param num_edges - The number of edges in the minpres.
 * @param num_points - The number of points.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @return A biperscan_linkage_result_t describing the linkage hierarchy.
 */
template <
    std::unsigned_integral index_t, std::unsigned_integral grade_t = index_t>
biperscan_linkage_result_t<index_t, grade_t> biperscan_linkage(
    grade_t const *lens_ptr, grade_t const *dist_ptr, index_t const *parent_ptr,
    index_t const *child_ptr, std::size_t const num_edges,
    std::size_t const num_points
) {
  auto const start{std::chrono::steady_clock::now()};
  linkage_hierarchy_t hierarchy{
      std::views::zip_transform(
          [](auto &&...args) { return bigrade_t{args...}; },
          std::span{lens_ptr, lens_ptr + num_edges},
          std::span{dist_ptr, dist_ptr + num_edges}
      ),
      std::views::zip_transform(
          [](auto &&...args) { return edge_t{args...}; },
          std::span{parent_ptr, parent_ptr + num_edges},
          std::span{child_ptr, child_ptr + num_edges}
      ),
      num_points
  };
  auto const finish{std::chrono::steady_clock::now()};

  return biperscan_linkage_result_t{
      hierarchy.take_lens_grades(),
      hierarchy.take_distance_grades(),
      hierarchy.take_parents(),
      hierarchy.take_children(),
      hierarchy.take_parent_roots(),
      hierarchy.take_child_roots(),
      std::chrono::duration<double>(finish - start).count()  // seconds
  };
}

/**
 * @brief Python / Cython wrapper for minmax_of (std::minmax_element).
 * @tparam value_t   The value type.
 * @param values_ptr   [In] A pointer to the value range.
 * @param num_values   [In] The number of values in the range.
 * @return The minimum and maximum value in the range.
 */
template <typename value_t>
std::pair<value_t, value_t> minmax_of(
    value_t *values_ptr, std::size_t num_values
) {
  return bppc::minmax_of(std::span{values_ptr, values_ptr + num_values});
}
}  // namespace bppc::python

#endif  // BIPERSCAN_PYTHON_API_H
