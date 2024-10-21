#ifndef BIPERSCAN_LIB_BIPERSCAN_H
#define BIPERSCAN_LIB_BIPERSCAN_H

#include "lib/algorithm.h"
#include "lib/linkage_hierarchy.h"
#include "lib/minimal_presentation_merges.h"

namespace bppc {

/**
 * @brief Return type of `biperscan_linkage` containing the linkage hierarchy,
 * minimal presentation, and supporting vectors.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 */
template <std::unsigned_integral index_t, std::unsigned_integral grade_t>
struct biperscan_minpres_result_t {
  std::vector<index_t> col_to_edge;
  std::vector<index_t> row_to_point;
  std::vector<grade_t> lens_grades;
  minimal_presentation_t<index_t, grade_t> minimal_presentation;

  template <
      std::ranges::random_access_range dist_range_t,
      std::ranges::random_access_range lens_range_t>
    requires std::floating_point<std::ranges::range_value_t<dist_range_t>> &&
                 std::floating_point<std::ranges::range_value_t<lens_range_t>>
  biperscan_minpres_result_t(
      dist_range_t &&distances, lens_range_t &&point_lens
  )
      : col_to_edge(argsort_of<index_t>(std::forward<dist_range_t>(distances))),
        row_to_point(argsort_of<index_t>(point_lens)),
        lens_grades(dense_rank_from_argsort<grade_t>(
            std::forward<lens_range_t>(point_lens), this->row_to_point
        )),
        minimal_presentation(
            graded_matrix_t<index_t, grade_t>{
                lens_grades,
                ordinal_rank_from_argsort<grade_t>(this->col_to_edge),
                ordinal_rank_from_argsort<index_t>(this->row_to_point)
            },
            this->lens_grades.size()
        ) {}
};

/**
 * @brief Constructs a bi-graded minimal presentation from distances and point
 * lens values. Also returns several supporting vectors.
 * @param distances - [In] A range with data point distances.
 * @param point_lens - [In] A range with point-lens values.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @tparam dist_range_t - A sized_range container type over floating
 values.
 * @tparam lens_range_t - A sized_range container type over floating values.
 * @return A biperscan_minpres_result_t with the minimal presentation.
 */
template <
    std::unsigned_integral index_t, std::unsigned_integral grade_t = index_t,
    std::ranges::random_access_range dist_range_t,
    std::ranges::random_access_range lens_range_t>
  requires std::floating_point<std::ranges::range_value_t<dist_range_t>> &&
           std::floating_point<std::ranges::range_value_t<lens_range_t>>
biperscan_minpres_result_t<index_t, grade_t> biperscan_minpres(
    dist_range_t &&distances, lens_range_t &&point_lens
) {
  return biperscan_minpres_result_t<index_t, grade_t>{
      std::forward<dist_range_t>(distances),
      std::forward<lens_range_t>(point_lens)
  };
}

/**
 * @brief Extracts merges from a minimal presentation.
 * @param minpres - [In] The minimal presentation.
 * @param num_points - [In] The number of data points.
 * @param min_cluster_size - [In] The minimum cluster size.
 * @param limit_fraction - [In] The maximum distance grade fraction to use as
 * upper distance threshold.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @return A minimal_presentation_merges_t.
 */
template <
    std::unsigned_integral index_t, std::unsigned_integral grade_t = index_t>
minimal_presentation_merges_t<index_t, grade_t> biperscan_merges(
    minimal_presentation_t<index_t, grade_t> const &minpres,
    std::size_t const num_points, std::size_t const min_cluster_size = 10,
    double const limit_fraction = 1.0
) {
  return minimal_presentation_merges_t{
      minpres.grades(), minpres.edges(), num_points, min_cluster_size,
      limit_fraction
  };
}

/**
 * @brief Constructs a bi-graded linkage hierarchy from a minimal presentation.
 * @param minpres - [In] The minimal presentation.
 * @param num_points - [In] The number of data points.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @return A linkage_hierarchy_t.
 */
template <
    std::unsigned_integral index_t, std::unsigned_integral grade_t = index_t>
linkage_hierarchy_t<index_t, grade_t> biperscan_linkage(
    minimal_presentation_t<index_t, grade_t> const &minpres,
    std::size_t const num_points
) {
  return linkage_hierarchy_t{minpres.grades(), minpres.edges(), num_points};
}

}  // namespace bppc

#endif  // BIPERSCAN_LIB_BIPERSCAN_H