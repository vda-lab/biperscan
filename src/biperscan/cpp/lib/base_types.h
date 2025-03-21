#ifndef BIPERSCAN_LIB_BASE_TYPES_H
#define BIPERSCAN_LIB_BASE_TYPES_H

#include <concepts>
#include <algorithm>
#include <span>

namespace bppc {
// Don't pad between members in this file.
#pragma pack(push, 1)

/**
 * @brief A bigrade with distance and lens grades.
 * @tparam grade_t - Precision for storing unsigned grades.
 */
template <std::unsigned_integral grade_t>
struct bigrade_t {
  grade_t lens{};
  grade_t distance{};

  [[nodiscard]] bool operator==(bigrade_t const &) const = default;
};

/**
 * @brief An edge in a graded matrix or minimal presentation.
 * @tparam index_t - Precision for storing unsigned indices.
 */
template <std::unsigned_integral index_t>
struct edge_t {
  // Parent should be less than child.
  index_t parent{};
  index_t child{};

  [[nodiscard]] bool operator==(edge_t const &) const = default;
};

/**
 * @brief A link in a linkage hierarchy.
 * @tparam index_t - Precision for storing unsigned indices.
 */
template <std::unsigned_integral index_t>
struct link_t {
  // Parent root should be less than child root.
  index_t id{};
  index_t parent{};
  index_t child{};
  index_t parent_root{};
  index_t child_root{};

  [[nodiscard]] bool operator==(link_t const &) const = default;
};

/**
 * @brief A merge between clusters.
 * @tparam index_t - Precision for storing unsigned indices.
 */
template <std::unsigned_integral index_t>
struct merge_t {
  index_t start_column{};
  index_t end_column{};
  index_t parent{};
  index_t child{};
  std::span<index_t const> parent_side{};
  std::span<index_t const> child_side{};

  [[nodiscard]] bool operator==(merge_t const &other) const {
    return parent == other.parent and child == other.child and
           start_column == other.start_column and
           end_column == other.end_column and
           std::ranges::equal(parent_side, other.parent_side) and
           std::ranges::equal(child_side, other.child_side);
  }
};

#pragma pack(pop)
}  // namespace bppc

#endif  // BIPERSCAN_LIB_BASE_TYPES_H