#ifndef BIPERSCAN_LIB_ALGORITHM_H
#define BIPERSCAN_LIB_ALGORITHM_H

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "concepts.h"

namespace bppc {

template <
    std::unsigned_integral index_t, std::ranges::random_access_range range_t,
    typename comp_t = std::less<std::ranges::range_value_t<range_t>>>
[[nodiscard]] std::vector<index_t> argsort_of(
    range_t &&data,
    comp_t comp = std::less<std::ranges::range_value_t<range_t>>()
) {
  // Reserve for number of points
  auto const num_points = static_cast<index_t>(std::ranges::size(data));
  auto result = std::views::iota(index_t{0}, num_points) |
                std::ranges::to<std::vector<index_t>>();
  std::ranges::stable_sort(result, [&data, &comp](index_t a, index_t b) {
    return comp(data[a], data[b]);
  });
  // Return argsort vector
  return result;
}

template <
    std::unsigned_integral grade_t, std::ranges::random_access_range range_t>
  requires std::unsigned_integral<std::ranges::range_value_t<range_t>>
[[nodiscard]] std::vector<grade_t> ordinal_rank_from_argsort(range_t &&indices
) {
  std::size_t const num_points = std::ranges::size(indices);
  std::vector<grade_t> result(num_points);
  for (grade_t rank : std::views::iota(grade_t{0}, num_points))
    result[indices[rank]] = rank;
  return result;
}

template <
    std::unsigned_integral grade_t,
    std::ranges::random_access_range value_range_t,
    std::ranges::random_access_range index_range_t>
  requires std::unsigned_integral<std::ranges::range_value_t<index_range_t>>
[[nodiscard]] std::vector<grade_t> dense_rank_from_argsort(
    value_range_t &&values, index_range_t &&indices
) {
  // Allocate output
  using value_t = std::ranges::range_value_t<value_range_t>;
  using index_t = std::ranges::range_value_t<index_range_t>;
  std::vector<grade_t> dense_ranks(std::ranges::size(values));

  // Iterate over indices, increase counter every time we detect a new value
  grade_t num_unique_values = 0u;
  value_t prev_value = values[indices[0]];
  dense_ranks[indices[0]] = num_unique_values;
  for (index_t idx : indices | std::views::drop(1)) {
    if (value_t value = values[idx]; prev_value != value) {
      prev_value = value;
      ++num_unique_values;
    }
    dense_ranks[idx] = num_unique_values;
  }

  return dense_ranks;
}

template <std::ranges::forward_range range_t>
[[nodiscard]] std::pair<
    std::ranges::range_value_t<range_t>, std::ranges::range_value_t<range_t>>
minmax_of(range_t &&values) {
  auto [min, max] = std::ranges::minmax_element(std::forward<range_t>(values));
  return {*min, *max};
}
}  // namespace bppc

#endif  // BIPERSCAN_LIB_ALGORITHM_H
