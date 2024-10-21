#include <array>

#include "biperscan.h"
#include "doctest.h"
#include "npy.h"

using namespace bppc;

TEST_SUITE_BEGIN("minimal_presentation");

TEST_CASE("api") {
  std::vector<std::pair<bigrade_t<unsigned long>, edge_t<unsigned>>> edges{
      {{1ul, 6ul}, {0u, 1u}}, {{2ul, 5ul}, {1u, 2u}}, {{3ul, 4ul}, {0u, 3u}},
      {{4ul, 0ul}, {0u, 4u}}, {{4ul, 1ul}, {0u, 1u}}, {{4ul, 2ul}, {0u, 2u}},
      {{4ul, 3ul}, {0u, 3u}}
  };
  minimal_presentation_t minpres{std::from_range, edges};

  SUBCASE("edge_at") {
    CHECK(edge_t{0u, 1u} == minpres.edge_at({1ul, 6ul}));
    CHECK(edge_t{0u, 3u} == minpres.edge_at({3ul, 4ul}));
    CHECK(edge_t{0u, 3u} == minpres.edge_at({4ul, 3ul}));
  }

  SUBCASE("items") {
    CHECK(std::ranges::equal(edges, minpres.items()));
  }

  SUBCASE("grades") {
    CHECK(std::ranges::equal(std::views::elements<0>(edges), minpres.grades()));
  }

  SUBCASE("edges") {
    CHECK(std::ranges::equal(std::views::elements<1>(edges), minpres.edges()));
  }

  SUBCASE("allocations") {
    CHECK(minpres.size() == edges.size());
    CHECK_FALSE(minpres.empty());
    minpres.clear();
    CHECK(minpres.empty());
  }

  SUBCASE("take_lens_grades") {
    std::vector new_vec{minpres.take_lens_grades()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(edges, [](auto const &e) { return e.first.lens; })
    ));
  }

  SUBCASE("take_distance_grades") {
    std::vector new_vec{minpres.take_distance_grades()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     edges, [](auto const &e) { return e.first.distance; }
                 )
    ));
  }

  SUBCASE("take_parents") {
    std::vector new_vec{minpres.take_parents()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     edges, [](auto const &e) { return e.second.parent; }
                 )
    ));
  }

  SUBCASE("take_children") {
    std::vector new_vec{minpres.take_children()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     edges, [](auto const &e) { return e.second.child; }
                 )
    ));
  }
}

TEST_CASE("compute_minimal_presentation (small sample)") {
  // Configure input
  constexpr std::array distances{3., 4., 2., 1., 2., 4., 1., 3., 1., 1.};
  constexpr std::array lens{1., 1., 1., 1., 2.};
  std::vector row_to_point{argsort_of<unsigned>(lens)};
  minimal_presentation_t const actual{
      graded_matrix_t{
          dense_rank_from_argsort<unsigned>(lens, row_to_point),
          ordinal_rank_from_argsort<unsigned>(argsort_of<unsigned>(distances)),
          ordinal_rank_from_argsort<unsigned long>(row_to_point)
      },
      lens.size()
  };

  // Expected output
  minimal_presentation_t expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned long>>>{
          {{0ul, 4ul}, {0u, 3u}},
          {{0ul, 5ul}, {1u, 2u}},
          {{0ul, 6ul}, {0u, 1u}},
          {{1ul, 0ul}, {0u, 4u}},
          {{1ul, 1ul}, {0u, 1u}},
          {{1ul, 2ul}, {0u, 2u}},
          {{1ul, 3ul}, {0u, 3u}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("larger example (horse)") {
  // Configure input
  auto distances = npy::load<double>("../../tests/data/horse_distance.npy");
  auto lens = npy::load<double>("../../tests/data/horse_lens.npy");
  std::vector row_to_point{argsort_of<unsigned>(lens)};
  minimal_presentation_t const actual{
      graded_matrix_t{
          dense_rank_from_argsort<unsigned>(std::move(lens), row_to_point),
          ordinal_rank_from_argsort<unsigned>(argsort_of<unsigned>(distances)),
          ordinal_rank_from_argsort<unsigned long>(row_to_point)
      },
      distances.size()
  };
  CHECK(distances.size() > actual.size());

  // All columns filled and following the elder rule.
  for (auto [parent, child] : actual.edges())
    REQUIRE(parent < child);

  // Columns in lex order
  auto comp = bigrade_lex_less<unsigned>{};
  for (auto const &[prev, next] :
       std::views::zip(actual.grades(), actual.grades() | std::views::drop(1)))
    CHECK(comp(prev, next));
}

TEST_SUITE_END();
