#ifndef BIPERSCAN_LIB_PRIORITY_QUEUE_H
#define BIPERSCAN_LIB_PRIORITY_QUEUE_H

#include "base_types.h"

namespace bppc {

// ------ Bigrade Ordering types

/**
 * @brief std::less like type for finding all bigrades that exist at a
 * particular bigrade_t, i.e., have a lower grade for both lens and distance
 * dimensions.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @verbatim Lower than p?
 *   ^
 * t | F  F  F  F
 * s | F  F  F  F
 * i | F  F  p  F
 * d | T  T  F  F
 *   ------------->
 *        lens
 * @endverbatim
 */
template <std::unsigned_integral grade_t>
struct bigrade_less {
  constexpr bool operator()(
      bigrade_t<grade_t> const &lhs, bigrade_t<grade_t> const &rhs
  ) const {
    return lhs.lens < rhs.lens && lhs.distance < rhs.distance;
  }
};

/**
 * @brief std::greater like type for finding all bigrades that do not exist yet
 * at a particular bigrade_t, i.e., have a higher grade for both lens and
 * distance dimensions.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @verbatim Higher than p?
 *   ^
 * t | F  F  F  T
 * s | F  F  F  T
 * i | F  F  p  F
 * d | F  F  F  F
 *   ------------->
 *        lens
 * @endverbatim
 */
template <std::unsigned_integral grade_t>
struct bigrade_greater {
  constexpr bool operator()(
      bigrade_t<grade_t> const &lhs, bigrade_t<grade_t> const &rhs
  ) const {
    return lhs.lens > rhs.lens && lhs.distance > rhs.distance;
  }
};

/**
 * @brief std::less like type for ordering types lexicographically by
 * bigrade_t.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @verbatim Order when sorting with this type:
 *   ^
 * t | 4  8
 * s | 3  7
 * i | 2  6
 * d | 1  5
 *   ------->
 *     lens
 * @endverbatim
 */
template <std::unsigned_integral grade_t>
struct bigrade_lex_less {
  constexpr bool operator()(
      bigrade_t<grade_t> const &lhs, bigrade_t<grade_t> const &rhs
  ) const {
    return lhs.lens < rhs.lens ||
           (lhs.lens == rhs.lens && lhs.distance < rhs.distance);
  }
};

/**
 * @brief std::greater like type for ordering types lexicographically by
 * bigrade_t.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @verbatim Order when sorting with this type:
 *   ^
 * t | 5  1
 * s | 6  2
 * i | 7  3
 * d | 8  4
 *   ------->
 *     lens
 * @endverbatim
 */
template <std::unsigned_integral grade_t>
struct bigrade_lex_greater {
  constexpr bool operator()(
      bigrade_t<grade_t> const &lhs, bigrade_t<grade_t> const &rhs
  ) const {
    return lhs.lens > rhs.lens ||
           (lhs.lens == rhs.lens && lhs.distance > rhs.distance);
  }
};

/**
 * @brief std::less like type for ordering types co-lexicographically by
 * bigrade_t.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @verbatim Order when sorting with this type:
 *   ^
 * t | 7  8
 * s | 5  6
 * i | 3  4
 * d | 1  2
 *   ------->
 *     lens
 * @endverbatim
 */
template <std::unsigned_integral grade_t>
struct bigrade_colex_less {
  constexpr bool operator()(
      bigrade_t<grade_t> const &lhs, bigrade_t<grade_t> const &rhs
  ) const {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.lens < rhs.lens);
  }
};

/**
 * @brief std::greater like type for ordering types co-lexicographically by
 * bigrade_t.
 * @tparam grade_t - Precision for storing unsigned grades.
 * @verbatim Order when sorting with this type:
 *   ^
 * t | 2  1
 * s | 4  3
 * i | 6  5
 * d | 8  7
 *   ------->
 *     lens
 * @endverbatim
 */
template <std::unsigned_integral grade_t>
struct bigrade_colex_greater {
  constexpr bool operator()(
      bigrade_t<grade_t> const &lhs, bigrade_t<grade_t> const &rhs
  ) const {
    return lhs.distance > rhs.distance ||
           (lhs.distance == rhs.distance && lhs.lens > rhs.lens);
  }
};

}  // namespace bppc

#endif  // BIPERSCAN_LIB_PRIORITY_QUEUE_H
