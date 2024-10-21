#include <array>

#include "biperscan.h"
#include "doctest.h"
#include "lib/ostream.h"
#include "npy.h"

using namespace bppc;

template <typename type_t>
std::size_t count_union(std::span<type_t> a, std::span<type_t> b) {
  std::size_t cnt{0};
  auto a_it = a.begin();
  auto b_it = b.begin();
  while (a_it != a.end() and b_it != b.end()) {
    if (*a_it < *b_it) {
      ++a_it;
      continue;
    }
    if (*b_it < *a_it) {
      ++b_it;
      continue;
    }
    ++a_it;
    ++b_it;
    ++cnt;
  }
  return cnt;
}

TEST_SUITE_BEGIN("minimal_presentation_merges");

TEST_CASE("api") {
  std::vector<std::pair<bigrade_t<unsigned long>, edge_t<unsigned>>> edges{
      {{1ul, 6ul}, {0u, 1u}}, {{2ul, 5ul}, {1u, 2u}}, {{3ul, 4ul}, {0u, 3u}},
      {{4ul, 0ul}, {0u, 4u}}, {{4ul, 1ul}, {0u, 1u}}, {{4ul, 2ul}, {0u, 2u}},
      {{4ul, 3ul}, {0u, 3u}}
  };
  minimal_presentation_t minpres{std::from_range, edges};
  minimal_presentation_merges_t merges{
      minpres.grades(), minpres.edges(), 5ul, 2ul, 1.0
  };

  constexpr std::array side_one{1u, 2u};
  constexpr std::array side_two{0u, 2u, 4u};
  std::vector<std::pair<bigrade_t<unsigned long>, merge_t<unsigned>>>
      expected_merges{{{4ul, 0ul}, {1u, 0u, 1u, 5u, side_one, side_two}}};

  SUBCASE("items") {
    CHECK(std::ranges::equal(expected_merges, merges.items()));
  }

  SUBCASE("grades") {
    CHECK(std::ranges::equal(
        std::views::elements<0>(expected_merges), merges.grades()
    ));
  }

  SUBCASE("edges") {
    CHECK(std::ranges::equal(
        std::views::transform(
            expected_merges,
            [](auto const &item) {
              return edge_t{item.second.root_one, item.second.root_two};
            }
        ),
        merges.edges()
    ));
  }

  SUBCASE("merges") {
    CHECK(
        std::ranges::equal(std::views::values(expected_merges), merges.merges())
    );
  }

  SUBCASE("allocations") {
    CHECK(merges.size() == expected_merges.size());
    CHECK_FALSE(merges.empty());
    merges.clear();
    CHECK(merges.empty());
  }

  SUBCASE("take_lens_grades") {
    std::vector new_vec{merges.take_lens_grades()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     expected_merges, [](auto const &e) { return e.first.lens; }
                 )
    ));
  }

  SUBCASE("take_distance_grades") {
    std::vector new_vec{merges.take_distance_grades()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(
            expected_merges, [](auto const &e) { return e.first.distance; }
        )
    ));
  }

  SUBCASE("take_roots_one") {
    std::vector new_vec{merges.take_roots_one()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(
            expected_merges, [](auto const &e) { return e.second.root_one; }
        )
    ));
  }

  SUBCASE("take_roots_two") {
    std::vector new_vec{merges.take_roots_two()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(
            expected_merges, [](auto const &e) { return e.second.root_two; }
        )
    ));
  }

  SUBCASE("take_start_columns") {
    std::vector new_vec{merges.take_start_columns()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(
            expected_merges, [](auto const &e) { return e.second.start_column; }
        )
    ));
  }

  SUBCASE("take_end_columns") {
    std::vector new_vec{merges.take_end_columns()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(
            expected_merges, [](auto const &e) { return e.second.end_column; }
        )
    ));
  }

  SUBCASE("take_sides_one") {
    std::vector new_vec{merges.take_sides_one()};
    CHECK(new_vec.size() == expected_merges.size());
    for (auto const &[actual, expected] : std::views::zip(
             new_vec, std::views::transform(
                          expected_merges,
                          [](auto const &e) { return e.second.side_one; }
                      )
         ))
      REQUIRE(std::ranges::equal(actual, expected));
  }

  SUBCASE("take_sides_two") {
    std::vector new_vec{merges.take_sides_two()};
    CHECK(new_vec.size() == expected_merges.size());
    for (auto const &[actual, expected] : std::views::zip(
             new_vec, std::views::transform(
                          expected_merges,
                          [](auto const &e) { return e.second.side_two; }
                      )
         ))
      REQUIRE(std::ranges::equal(actual, expected));
  }
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
  minimal_presentation_merges_t merges{
      actual.grades(), actual.edges(), lens.size(), 25, 1.0
  };
  CHECK(merges.merges().size() == 479);

  std::size_t cnt{0};
  auto prev_end_column = 0u;
  auto prev_start_column = 0u;
  bigrade_t prev_grade{0u, 0u};
  for (auto const &[grade, merge] : merges.items()) {
    REQUIRE(merge.root_one != merge.root_two);
    REQUIRE(merge.end_column > merge.start_column);
    if (merge.start_column == prev_start_column and
        merge.end_column == prev_end_column) {
      if (cnt > 0)
        REQUIRE(prev_grade.lens == grade.lens);
      REQUIRE(prev_grade.distance < grade.distance);
      REQUIRE(merge.side_one.size() + merge.side_two.size() > 0);
    } else {
      REQUIRE(merge.side_one.size() >= 25);
      REQUIRE(merge.side_two.size() >= 25);
    }
    REQUIRE(count_union(merge.side_one, merge.side_two) <= 1);
    prev_end_column = merge.end_column;
    prev_start_column = merge.start_column;
    prev_grade = grade;
    ++cnt;
  }
}

TEST_SUITE_END();
