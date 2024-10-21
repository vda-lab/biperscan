#include <array>

#include "doctest.h"
#include "lib/algorithm.h"
using namespace bppc;

TEST_SUITE_BEGIN("algorithm");

TEST_CASE("argsort_of") {
  std::vector const expected{3u, 2u, 0u, 1u, 4u};
  SUBCASE("array") {
    constexpr std::array values{2u, 3u, 1u, 0u, 4u};
    CHECK(argsort_of<unsigned>(values) == expected);
  }
  SUBCASE("vector") {
    std::vector const values{2., 3., 1., 0., 4.};
    CHECK(argsort_of<unsigned>(values) == expected);
  }
  SUBCASE("comp") {
    constexpr std::array values{2l, 3l, 1l, 0l, 4l};
    std::vector const reversed{4u, 1u, 0u, 2u, 3u};
    CHECK(argsort_of<unsigned>(values, std::greater<>()) == reversed);
  }
}

TEST_CASE("ordinal_rank_from_argsort") {
  std::vector expected{2u, 3u, 1u, 0u, 4u};
  SUBCASE("array") {
    constexpr std::array values{3u, 2u, 0u, 1u, 4u};
    CHECK(ordinal_rank_from_argsort<unsigned>(values) == expected);
  }
  SUBCASE("vector") {
    std::vector values{3u, 2u, 0u, 1u, 4u};
    CHECK(ordinal_rank_from_argsort<unsigned>(std::move(values)) == expected);
  }
}

TEST_CASE("dense_rank_from_argsort") {
  constexpr std::array values{2u, 3u, 0u, 0u, 4u};
  std::vector expected{1u, 2u, 0u, 0u, 3u};

  SUBCASE("array") {
    constexpr std::array indices{2u, 3u, 0u, 1u, 4u};
    CHECK(dense_rank_from_argsort<unsigned>(values, indices) == expected);
  }
  SUBCASE("vector") {
    std::vector indices{2u, 3u, 0u, 1u, 4u};
    CHECK(
        dense_rank_from_argsort<unsigned>(values, std::move(indices)) ==
        expected
    );
  }
}

TEST_CASE("minmax_of") {
  constexpr std::pair expected{-5, 10};
  SUBCASE("array even") {
    constexpr std::array values{2, -5, 10, 0};
    CHECK(minmax_of(values) == expected);
  }
  SUBCASE("array odd") {
    constexpr std::array values{2, -5, 0, 3, 10};
    CHECK(minmax_of(values) == expected);
  }
  SUBCASE("vector even") {
    std::vector const values{2, -5, 10, 0};
    CHECK(minmax_of(values) == expected);
  }
  SUBCASE("vector odd") {
    std::vector const values{2, -5, 0, 3, 10};
    CHECK(minmax_of(values) == expected);
  }
}

TEST_SUITE_END();