#include <array>

#include "doctest.h"
#include "lib/algorithm.h"
#include "lib/graded_matrix.h"
#include "lib/ostream.h"
#include "npy.h"

using namespace bppc;

TEST_SUITE_BEGIN("graded_matrix");

TEST_CASE("api") {
  std::vector<std::pair<bigrade_t<unsigned long>, edge_t<unsigned>>> edges{
      {{1ul, 6ul}, {0u, 1u}}, {{2ul, 5ul}, {1u, 2u}}, {{2ul, 8ul}, {0u, 2u}},
      {{3ul, 4ul}, {0u, 3u}}, {{3ul, 7ul}, {2u, 3u}}, {{3ul, 9ul}, {1u, 3u}},
      {{4ul, 0ul}, {0u, 4u}}, {{4ul, 1ul}, {1u, 4u}}, {{4ul, 2ul}, {2u, 4u}},
      {{4ul, 3ul}, {3u, 4u}}
  };
  graded_matrix_t matrix{std::from_range, edges};

  SUBCASE("items") {
    CHECK(std::ranges::equal(edges, matrix.items()));
    graded_matrix_t const copy = matrix;
    CHECK(std::ranges::equal(edges, copy.items()));
  }

  SUBCASE("grades") {
    CHECK(std::ranges::equal(std::views::elements<0>(edges), matrix.grades()));
  }

  SUBCASE("edges") {
    CHECK(std::ranges::equal(std::views::elements<1>(edges), matrix.edges()));
  }

  SUBCASE("allocations") {
    CHECK(matrix.size() == edges.size());
    CHECK_FALSE(matrix.empty());
    matrix.clear();
    CHECK(matrix.empty());
  }
}

TEST_CASE("construct_graded_matrix") {
  // Input
  constexpr std::array distances{3., 4., 2., 1., 2., 4., 1., 3., 1., 1.};
  constexpr std::array lens{1., 1., 1., 1., 2.};
  std::vector row_to_point{argsort_of<unsigned>(lens)};
  graded_matrix_t actual{
      dense_rank_from_argsort<unsigned long>(lens, row_to_point),
      ordinal_rank_from_argsort<unsigned long>(argsort_of<unsigned>(distances)),
      ordinal_rank_from_argsort<unsigned>(row_to_point)
  };

  // Expected
  graded_matrix_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned long>, edge_t<unsigned>>>{
          {{0ul, 4ul}, {0u, 3u}},
          {{0ul, 5ul}, {1u, 2u}},
          {{0ul, 6ul}, {0u, 1u}},
          {{0ul, 7ul}, {2u, 3u}},
          {{0ul, 8ul}, {0u, 2u}},
          {{0ul, 9ul}, {1u, 3u}},
          {{1ul, 0ul}, {0u, 4u}},
          {{1ul, 1ul}, {1u, 4u}},
          {{1ul, 2ul}, {2u, 4u}},
          {{1ul, 3ul}, {3u, 4u}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("construct_graded_matrix (horse)") {
  std::vector<double> distances =
      npy::load<double>("../../tests/data/horse_distance.npy");
  std::vector<double> lens = npy::load<double>("../../tests/data/horse_lens.npy");
  std::vector row_to_point{argsort_of<unsigned>(lens)};
  graded_matrix_t actual{
      dense_rank_from_argsort<unsigned long>(std::move(lens), row_to_point),
      ordinal_rank_from_argsort<unsigned long>(argsort_of<unsigned>(distances)),
      ordinal_rank_from_argsort<unsigned>(row_to_point)
  };
  CHECK(actual.size() == distances.size());

  // Row to point is increasing lens order
  double prev_lens = lens[row_to_point[0]];
  for (unsigned point : row_to_point | std::views::drop(1)) {
    double value = lens[point];
    REQUIRE(value >= prev_lens);
    prev_lens = value;
  }

  // All columns filled and following the elder rule.
  for (auto const &[parent, child] : actual.edges())
    REQUIRE(parent < child);
}

TEST_SUITE_END();
