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
      {{1ul, 15ul}, {0u, 1u}}, {{2ul, 1ul}, {0u, 2u}}, {{2ul, 2ul}, {1u, 3u}},
      {{2ul, 10ul}, {0u, 1u}}, {{3ul, 5ul}, {3u, 4u}}, {{3ul, 7ul}, {0u, 1u}},
      {{4ul, 3ul}, {4u, 5u}},  {{4ul, 4ul}, {2u, 4u}}
  };
  minimal_presentation_t minpres{std::from_range, edges};
  minimal_presentation_merges_t merges{
      minpres.grades(), minpres.edges(), 6ul, 2ul, 1.0
  };

  // Need to exist for the lifetime of the expected_merges object!
  constexpr std::array parent_side {0u, 2u};
  constexpr std::array child_side {1u, 3u, 4u};
  std::vector<std::pair<bigrade_t<unsigned long>, merge_t<unsigned>>>
      expected_merges{
          {{2ul, 10ul}, {0u, 3u, 0u, 1u, parent_side, std::span{child_side.begin(), child_side.end()-1}}},
          {{3ul,  7ul}, {3u, 5u, 0u, 1u, parent_side, std::span{child_side.begin(), child_side.end()}}},
          {{4ul,  5ul}, {4u, 7u, 0u, 1u, parent_side, std::span{child_side.begin(), child_side.end()-1}}},
      };

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
              return edge_t{item.second.parent, item.second.child};
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
    std::vector new_vec{merges.take_parents()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(
            expected_merges, [](auto const &e) { return e.second.parent; }
        )
    ));
  }

  SUBCASE("take_roots_two") {
    std::vector new_vec{merges.take_children()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(
            expected_merges, [](auto const &e) { return e.second.child; }
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

  SUBCASE("take_parent_side") {
    std::vector new_vec{merges.take_parent_sides()};
    CHECK(new_vec.size() == expected_merges.size());
    for (auto const &[actual, expected] : std::views::zip(
             new_vec, std::views::transform(
                          expected_merges,
                          [](auto const &e) { return e.second.parent_side; }
                      )
         ))
      REQUIRE(std::ranges::equal(actual, expected));
  }

  SUBCASE("take_child_side") {
    std::vector new_vec{merges.take_child_sides()};
    CHECK(new_vec.size() == expected_merges.size());
    for (auto const &[actual, expected] : std::views::zip(
             new_vec, std::views::transform(
                          expected_merges,
                          [](auto const &e) { return e.second.child_side; }
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
  CHECK(merges.merges().size() == 51);

  std::size_t cnt{0};
  auto prev_column = std::pair(0u, 0u);
  for (auto const &[grade, merge] : merges.items()) {
    auto const column = std::pair(merge.start_column, merge.end_column);
    REQUIRE(column != prev_column);
    prev_column = column;
    REQUIRE(merge.parent != merge.child);
    REQUIRE(merge.end_column > merge.start_column);
    REQUIRE(merge.parent_side.size() >= 25);
    REQUIRE(merge.child_side.size() >= 25);
    REQUIRE(count_union(merge.parent_side, merge.child_side) == 0);
    auto it = std::ranges::lower_bound(merge.parent_side, merge.parent);
    REQUIRE(it != merge.parent_side.end());
    REQUIRE(*it == merge.parent);
    it = std::ranges::lower_bound(merge.child_side, merge.child);
    REQUIRE(it != merge.child_side.end());
    REQUIRE(*it == merge.child);
  }
}

TEST_SUITE_END();
