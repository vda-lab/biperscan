
#include "biperscan.h"
#include "doctest.h"
#include "lib/ostream.h"
#include "npy.h"

using namespace bppc;

TEST_SUITE_BEGIN("linkage hierarchy");

TEST_CASE("api") {
  std::vector<std::pair<bigrade_t<unsigned long>, link_t<unsigned>>> const
      links{{{0, 4}, {5, 0, 3, 0, 3}},  {{0, 5}, {6, 1, 2, 1, 2}},
            {{0, 6}, {7, 5, 6, 0, 1}},  {{1, 0}, {8, 0, 4, 0, 4}},
            {{1, 1}, {9, 8, 1, 0, 1}},  {{1, 2}, {10, 9, 2, 0, 2}},
            {{1, 3}, {11, 10, 3, 0, 3}}};
  unsigned const num_points = links[0].second.id;
  linkage_hierarchy_t hierarchy{std::from_range, links};

  SUBCASE("link_at") {
    CHECK(link_t{5u, 0u, 3u, 0u, 3u} == hierarchy.link_at({0ul, 4ul}));
    CHECK(link_t{8u, 0u, 4u, 0u, 4u} == hierarchy.link_at({1ul, 0ul}));
    CHECK(link_t{11u, 10u, 3u, 0u, 3u} == hierarchy.link_at({1ul, 3ul}));
  }

  SUBCASE("items") {
    CHECK(std::ranges::equal(links, hierarchy.items()));
  }

  SUBCASE("grades") {
    CHECK(std::ranges::equal(std::views::elements<0>(links), hierarchy.grades())
    );
  }

  SUBCASE("links") {
    CHECK(std::ranges::equal(std::views::elements<1>(links), hierarchy.links())
    );
  }

  SUBCASE("is_link") {
    for (unsigned i = 0; i < num_points; ++i) {
      REQUIRE_FALSE(hierarchy.is_link(i));
    }
    for (unsigned i = num_points; i < num_points + links.size(); ++i) {
      REQUIRE(hierarchy.is_link(i));
    }
  }

  SUBCASE("lens_grade_of") {
    for (unsigned idx = 0; idx < links.size(); ++idx) {
      REQUIRE(
          hierarchy.lens_grade_of(idx + num_points) == links[idx].first.lens
      );
    }
  }

  SUBCASE("distance_grade_of") {
    for (unsigned idx = 0; idx < links.size(); ++idx) {
      REQUIRE(
          hierarchy.distance_grade_of(idx + num_points) ==
          links[idx].first.distance
      );
    }
  }

  SUBCASE("grade_of") {
    for (unsigned idx = 0; idx < links.size(); ++idx) {
      REQUIRE(hierarchy.grade_of(idx + num_points) == links[idx].first);
    }
  }

  SUBCASE("parent_of") {
    for (unsigned idx = 0; idx < links.size(); ++idx) {
      REQUIRE(
          hierarchy.parent_of(idx + num_points) == links[idx].second.parent
      );
    }
  }

  SUBCASE("child_of") {
    for (unsigned idx = 0; idx < links.size(); ++idx) {
      REQUIRE(hierarchy.child_of(idx + num_points) == links[idx].second.child);
    }
  }

  SUBCASE("parent_root_of") {
    for (unsigned idx = 0; idx < links.size(); ++idx) {
      REQUIRE(
          hierarchy.parent_root_of(idx + num_points) ==
          links[idx].second.parent_root
      );
    }
  }

  SUBCASE("child_root_of") {
    for (unsigned idx = 0; idx < links.size(); ++idx) {
      REQUIRE(
          hierarchy.child_root_of(idx + num_points) ==
          links[idx].second.child_root
      );
    }
  }

  SUBCASE("link_of") {
    for (unsigned idx = 0; idx < links.size(); ++idx) {
      auto link = hierarchy.link_of(idx + num_points);
      REQUIRE(link.id == links[idx].second.id);
      REQUIRE(link.parent == links[idx].second.parent);
      REQUIRE(link.child == links[idx].second.child);
      REQUIRE(link.parent_root == links[idx].second.parent_root);
      REQUIRE(link.child_root == links[idx].second.child_root);
    }
  }

  SUBCASE("take_lens_grades") {
    std::vector new_vec{hierarchy.take_lens_grades()};
    CHECK(std::ranges::equal(
        new_vec,
        std::views::transform(links, [](auto const &l) { return l.first.lens; })
    ));
  }

  SUBCASE("take_distance_grades") {
    std::vector new_vec{hierarchy.take_distance_grades()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     links, [](auto const &l) { return l.first.distance; }
                 )
    ));
  }

  SUBCASE("take_parents") {
    std::vector new_vec{hierarchy.take_parents()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     links, [](auto const &l) { return l.second.parent; }
                 )
    ));
  }

  SUBCASE("take_children") {
    std::vector new_vec{hierarchy.take_children()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     links, [](auto const &l) { return l.second.child; }
                 )
    ));
  }

  SUBCASE("take_parent_roots") {
    std::vector new_vec{hierarchy.take_parent_roots()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     links, [](auto const &l) { return l.second.parent_root; }
                 )
    ));
  }

  SUBCASE("take_child_roots") {
    std::vector new_vec{hierarchy.take_child_roots()};
    CHECK(std::ranges::equal(
        new_vec, std::views::transform(
                     links, [](auto const &l) { return l.second.child_root; }
                 )
    ));
  }
}

TEST_CASE("empty minpres") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{}
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 0u};

  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{}
  };
  CHECK(actual == expected);
}

TEST_CASE("small sample") {
  // Configure input
  constexpr std::array lens{1., 1., 1., 1., 2.};
  constexpr std::array distances{3., 4., 2., 1., 2., 4., 1., 3., 1., 1.};
  std::vector row_to_point{argsort_of<unsigned>(lens)};
  minimal_presentation_t const minpres{
      graded_matrix_t{
          dense_rank_from_argsort<unsigned>(lens, row_to_point),
          ordinal_rank_from_argsort<unsigned>(argsort_of<unsigned>(distances)),
          ordinal_rank_from_argsort<unsigned long>(row_to_point)
      },
      lens.size()
  };
  linkage_hierarchy_t const actual{
      minpres.grades(), minpres.edges(), lens.size()
  };

  // Expected value
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned long>>>{
          {{0, 4}, {5, 0, 3, 0, 3}},
          {{0, 5}, {6, 1, 2, 1, 2}},
          {{0, 6}, {7, 5, 6, 0, 1}},
          {{1, 0}, {8, 0, 4, 0, 4}},
          {{1, 1}, {9, 8, 1, 0, 1}},
          {{1, 2}, {10, 9, 2, 0, 2}},
          {{1, 3}, {11, 10, 3, 0, 3}}
      }
  };
  // Actual values
  CHECK(actual == expected);
}

TEST_CASE("multi-column duplicates (1)") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{0, 3}, {1, 2}}, {{1, 1}, {1, 3}}, {{2, 0}, {3, 4}}, {{2, 2}, {0, 1}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 5u};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{0, 3}, {5, 1, 2, 1, 2}},
          {{1, 1}, {6, 1, 3, 1, 3}},
          {{1, 3}, {7, 5, 6, 1, 1}},
          {{2, 0}, {8, 3, 4, 3, 4}},
          {{2, 1}, {9, 6, 8, 1, 3}},
          {{2, 2}, {10, 0, 9, 0, 1}},
          {{2, 3}, {11, 10, 7, 0, 1}},
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("multi-column duplicates (2)") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{0, 3}, {1, 2}}, {{1, 2}, {1, 3}}, {{2, 0}, {3, 4}}, {{2, 1}, {0, 1}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 5u};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{0, 3}, {5, 1, 2, 1, 2}},
          {{1, 2}, {6, 1, 3, 1, 3}},
          {{1, 3}, {7, 5, 6, 1, 1}},
          {{2, 0}, {8, 3, 4, 3, 4}},
          {{2, 1}, {9, 0, 1, 0, 1}},
          {{2, 2}, {10, 6, 8, 1, 3}},
          {{2, 2}, {11, 9, 10, 0, 1}},
          {{2, 3}, {12, 11, 7, 0, 1}},
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("multi-column duplicates (3)") {
  constexpr std::array distances{
      0.39345239, 0.77746905, 0.4493711,  0.57992826, 0.44099655, 0.5017199,
      0.08906789, 0.35487395, 0.91390041, 0.75822986, 0.09804144, 0.39633745,
      0.61092208, 0.76284835, 0.47468416, 0.57699653, 1.0,        0.70636699,
      0.65119059, 0.43041133, 0.54269353, 0.78057259, 0.62904309, 0.30233601,
      0.41138111, 0.61491897, 0.76668761, 0.52275092, 0.61526657, 0.9597984,
      0.54405431, 0.77067935, 0.64779527, 0.49926722, 0.89988507, 0.23540753,
      0.42605228, 0.22077842, 0.51196866, 0.4459976,  0.36511186, 0.50970468,
      0.35268286, 0.88865498, 0.72317087
  };
  constexpr std::array lens{0.22358826, 0., 0.56982139, 0.10439777,
                            0.25998408, 1., 0.59551271, 0.28914699,
                            0.7188854,  0.};
  std::vector row_to_point{argsort_of<unsigned>(lens)};
  minimal_presentation_t const minpres{
      graded_matrix_t{
          dense_rank_from_argsort<unsigned>(lens, row_to_point),
          ordinal_rank_from_argsort<unsigned>(argsort_of<unsigned>(distances)),
          ordinal_rank_from_argsort<unsigned>(row_to_point)
      },
      lens.size()
  };

  linkage_hierarchy_t const actual{
      minpres.grades(), minpres.edges(), lens.size()
  };
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{0, 44}, {10, 0, 1, 0, 1}},   {{1, 1}, {11, 0, 2, 0, 2}},
          {{1, 43}, {12, 11, 1, 0, 1}},  {{2, 8}, {13, 11, 3, 0, 3}},
          {{2, 42}, {14, 13, 1, 0, 1}},  {{3, 9}, {15, 13, 4, 0, 4}},
          {{3, 41}, {16, 15, 1, 0, 1}},  {{4, 0}, {17, 3, 5, 3, 5}},
          {{4, 8}, {18, 13, 17, 0, 3}},  {{4, 9}, {19, 15, 18, 0, 0}},
          {{4, 40}, {20, 19, 1, 0, 1}},  {{5, 4}, {21, 1, 6, 1, 6}},
          {{5, 31}, {22, 19, 21, 0, 1}}, {{6, 14}, {23, 19, 7, 0, 7}},
          {{6, 19}, {24, 23, 21, 0, 1}}, {{7, 5}, {25, 17, 8, 3, 8}},
          {{7, 7}, {26, 25, 7, 3, 7}},   {{7, 8}, {27, 18, 26, 0, 3}},
          {{7, 9}, {28, 19, 27, 0, 0}},  {{7, 19}, {29, 24, 28, 0, 0}},
          {{8, 2}, {30, 8, 9, 8, 9}},    {{8, 3}, {31, 7, 30, 7, 8}},
          {{8, 5}, {32, 25, 31, 3, 7}},  {{8, 8}, {33, 27, 32, 0, 3}},
          {{8, 9}, {34, 28, 33, 0, 0}},  {{8, 12}, {35, 34, 21, 0, 1}}
      }
  };

  CHECK(actual == expected);
}

TEST_CASE("avoid zero links (0)") {
  constexpr std::size_t num_points = 7;
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{0, 3}, {1, 2}},
          {{0, 5}, {0, 1}},
          {{1, 4}, {2, 3}},
          {{2, 6}, {4, 5}},
          {{3, 7}, {0, 4}},
          {{4, 1}, {2, 6}},
          {{4, 2}, {2, 5}},
          {{5, 0}, {5, 6}}
      }
  };
  linkage_hierarchy_t const actual{
      minpres.grades(), minpres.edges(), num_points
  };

  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{0, 3}, {7, 1, 2, 1, 2}},
          {{0, 5}, {8, 0, 7, 0, 1}},
          {{1, 4}, {9, 7, 3, 1, 3}},
          {{1, 5}, {10, 8, 9, 0, 1}},
          {{2, 6}, {11, 4, 5, 4, 5}},
          {{3, 7}, {12, 10, 11, 0, 4}},
          {{4, 1}, {13, 2, 6, 2, 6}},
          {{4, 2}, {14, 13, 5, 2, 5}},
          {{4, 3}, {15, 7, 14, 1, 2}},
          {{4, 4}, {16, 9, 15, 1, 1}},
          {{4, 5}, {17, 10, 16, 0, 1}},
          {{4, 6}, {18, 17, 11, 0, 4}},
          {{5, 0}, {19, 5, 6, 5, 6}},
          {{5, 1}, {20, 13, 19, 2, 5}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("avoid zero links (1)") {
  // Random ordered minpres edges (grade, parent, child)
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{18, 33}, {0, 8}},   {{0, 32}, {0, 1}},    {{1, 31}, {1, 2}},
          {{2, 30}, {2, 3}},    {{7, 29}, {8, 9}},    {{8, 28}, {8, 10}},
          {{19, 27}, {8, 16}},  {{11, 26}, {8, 9}},   {{9, 25}, {8, 10}},
          {{20, 24}, {9, 16}},  {{25, 23}, {3, 9}},   {{3, 22}, {3, 4}},
          {{11, 21}, {9, 12}},  {{21, 20}, {16, 17}}, {{22, 19}, {10, 17}},
          {{4, 18}, {4, 5}},    {{23, 17}, {10, 17}}, {{24, 16}, {10, 17}},
          {{15, 15}, {10, 12}}, {{13, 14}, {12, 14}}, {{16, 13}, {10, 12}},
          {{10, 12}, {10, 11}}, {{12, 11}, {12, 13}}, {{5, 10}, {5, 6}},
          {{6, 9}, {6, 7}},     {{26, 8}, {7, 11}},   {{17, 7}, {11, 13}},
          {{23, 5}, {17, 18}},  {{27, 6}, {13, 17}},  {{14, 4}, {13, 14}},
          {{14, 3}, {14, 15}},  {{28, 2}, {15, 18}},  {{28, 1}, {15, 19}},
          {{29, 0}, {18, 19}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 20u};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{0, 32}, {20, 0, 1, 0, 1}},       {{1, 31}, {21, 1, 2, 1, 2}},
          {{1, 32}, {22, 20, 21, 0, 1}},     {{2, 30}, {23, 2, 3, 2, 3}},
          {{2, 31}, {24, 21, 23, 1, 2}},     {{2, 32}, {25, 22, 24, 0, 1}},
          {{3, 22}, {26, 3, 4, 3, 4}},       {{3, 30}, {27, 23, 26, 2, 3}},
          {{3, 31}, {28, 24, 27, 1, 2}},     {{3, 32}, {29, 25, 28, 0, 1}},
          {{4, 18}, {30, 4, 5, 4, 5}},       {{4, 22}, {31, 26, 30, 3, 4}},
          {{4, 30}, {32, 27, 31, 2, 3}},     {{4, 31}, {33, 28, 32, 1, 2}},
          {{4, 32}, {34, 29, 33, 0, 1}},     {{5, 10}, {35, 5, 6, 5, 6}},
          {{5, 18}, {36, 30, 35, 4, 5}},     {{5, 22}, {37, 31, 36, 3, 4}},
          {{5, 30}, {38, 32, 37, 2, 3}},     {{5, 31}, {39, 33, 38, 1, 2}},
          {{5, 32}, {40, 34, 39, 0, 1}},     {{6, 9}, {41, 6, 7, 6, 7}},
          {{6, 10}, {42, 35, 41, 5, 6}},     {{6, 18}, {43, 36, 42, 4, 5}},
          {{6, 22}, {44, 37, 43, 3, 4}},     {{6, 30}, {45, 38, 44, 2, 3}},
          {{6, 31}, {46, 39, 45, 1, 2}},     {{6, 32}, {47, 40, 46, 0, 1}},
          {{7, 29}, {48, 8, 9, 8, 9}},       {{8, 28}, {49, 8, 10, 8, 10}},
          {{8, 29}, {50, 48, 49, 8, 8}},     {{9, 25}, {51, 8, 10, 8, 10}},
          {{10, 12}, {52, 10, 11, 10, 11}},  {{10, 25}, {53, 51, 52, 8, 10}},
          {{10, 29}, {54, 50, 53, 8, 8}},    {{11, 21}, {55, 9, 12, 9, 12}},
          {{11, 26}, {56, 53, 55, 8, 9}},    {{12, 11}, {57, 12, 13, 12, 13}},
          {{12, 21}, {58, 55, 57, 9, 12}},   {{12, 26}, {59, 56, 58, 8, 9}},
          {{13, 14}, {60, 57, 14, 12, 14}},  {{13, 21}, {61, 58, 60, 9, 12}},
          {{13, 26}, {62, 59, 61, 8, 9}},    {{14, 3}, {63, 14, 15, 14, 15}},
          {{14, 4}, {64, 13, 63, 13, 14}},   {{14, 11}, {65, 57, 64, 12, 13}},
          {{14, 21}, {66, 61, 65, 9, 12}},   {{14, 26}, {67, 62, 66, 8, 9}},
          {{15, 15}, {68, 52, 65, 10, 12}},  {{15, 21}, {69, 66, 68, 9, 10}},
          {{15, 25}, {70, 53, 69, 8, 9}},    {{16, 13}, {71, 52, 65, 10, 12}},
          {{17, 7}, {72, 11, 64, 11, 13}},   {{17, 11}, {73, 72, 65, 11, 12}},
          {{17, 12}, {74, 52, 73, 10, 11}},  {{18, 33}, {75, 47, 70, 0, 8}},
          {{19, 27}, {76, 70, 16, 8, 16}},   {{19, 33}, {77, 75, 76, 0, 8}},
          {{20, 24}, {78, 69, 16, 9, 16}},   {{20, 25}, {79, 70, 78, 8, 9}},
          {{21, 20}, {80, 16, 17, 16, 17}},  {{21, 24}, {81, 78, 80, 9, 16}},
          {{21, 25}, {82, 79, 81, 8, 9}},    {{21, 33}, {83, 77, 82, 0, 8}},
          {{22, 19}, {84, 74, 17, 10, 17}},  {{22, 20}, {85, 84, 80, 10, 16}},
          {{22, 21}, {86, 69, 85, 9, 10}},   {{23, 5}, {87, 17, 18, 17, 18}},
          {{23, 17}, {88, 74, 87, 10, 17}},  {{23, 20}, {89, 85, 88, 10, 10}},
          {{23, 21}, {90, 86, 89, 9, 10}},   {{23, 25}, {91, 82, 90, 8, 9}},
          {{23, 33}, {92, 83, 91, 0, 8}},    {{24, 16}, {93, 74, 87, 10, 17}},
          {{25, 23}, {94, 44, 90, 3, 9}},    {{25, 25}, {95, 94, 91, 3, 8}},
          {{25, 30}, {96, 45, 95, 2, 3}},    {{25, 31}, {97, 46, 96, 1, 2}},
          {{25, 32}, {98, 47, 97, 0, 1}},    {{26, 8}, {99, 7, 72, 7, 11}},
          {{26, 9}, {100, 41, 99, 6, 7}},    {{26, 10}, {101, 42, 100, 5, 6}},
          {{26, 11}, {102, 101, 73, 5, 11}}, {{26, 12}, {103, 102, 74, 5, 10}},
          {{26, 16}, {104, 103, 93, 5, 10}}, {{26, 18}, {105, 43, 104, 4, 5}},
          {{26, 20}, {106, 105, 89, 4, 10}}, {{26, 21}, {107, 106, 90, 4, 9}},
          {{26, 22}, {108, 44, 107, 3, 4}},  {{27, 6}, {109, 64, 87, 13, 17}},
          {{27, 7}, {110, 72, 109, 11, 13}}, {{27, 8}, {111, 99, 110, 7, 11}},
          {{27, 9}, {112, 100, 111, 6, 7}},  {{27, 10}, {113, 101, 112, 5, 6}},
          {{27, 11}, {114, 102, 113, 5, 5}}, {{27, 12}, {115, 103, 114, 5, 5}},
          {{28, 1}, {116, 15, 19, 15, 19}},  {{28, 2}, {117, 116, 18, 15, 18}},
          {{28, 3}, {118, 63, 117, 14, 15}}, {{28, 4}, {119, 64, 118, 13, 14}},
          {{28, 5}, {120, 119, 87, 13, 17}}, {{28, 7}, {121, 110, 120, 11, 13}},
          {{28, 8}, {122, 111, 121, 7, 11}}, {{28, 9}, {123, 112, 122, 6, 7}},
          {{28, 10}, {124, 113, 123, 5, 6}}, {{28, 11}, {125, 114, 124, 5, 5}},
          {{28, 12}, {126, 115, 125, 5, 5}}, {{28, 18}, {127, 105, 126, 4, 5}},
          {{28, 20}, {128, 106, 127, 4, 4}}, {{28, 21}, {129, 107, 128, 4, 4}},
          {{28, 22}, {130, 108, 129, 3, 4}}, {{28, 25}, {131, 95, 130, 3, 3}},
          {{28, 30}, {132, 96, 131, 2, 3}},  {{28, 31}, {133, 97, 132, 1, 2}},
          {{28, 32}, {134, 98, 133, 0, 1}},  {{29, 0}, {135, 18, 19, 18, 19}},
          {{29, 1}, {136, 116, 135, 15, 18}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("avoid zero links (2)") {
  // Random ordered minpres edges (grade, parent, child)
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{33, 0}, {19, 25}},  {{19, 1}, {18, 19}},  {{18, 2}, {17, 18}},
          {{17, 3}, {16, 17}},  {{38, 4}, {12, 16}},  {{16, 5}, {13, 16}},
          {{11, 6}, {11, 12}},  {{14, 7}, {10, 13}},  {{32, 8}, {10, 25}},
          {{29, 9}, {24, 25}},  {{30, 10}, {23, 24}}, {{26, 11}, {22, 24}},
          {{8, 12}, {9, 10}},   {{29, 13}, {22, 23}}, {{10, 14}, {6, 11}},
          {{6, 15}, {8, 9}},    {{37, 16}, {6, 8}},   {{26, 17}, {22, 23}},
          {{5, 18}, {7, 8}},    {{25, 19}, {21, 23}}, {{31, 20}, {7, 21}},
          {{3, 21}, {4, 6}},    {{2, 22}, {2, 4}},    {{36, 23}, {2, 7}},
          {{35, 24}, {2, 7}},   {{12, 25}, {3, 7}},   {{34, 26}, {2, 3}},
          {{28, 27}, {3, 21}},  {{4, 28}, {5, 7}},    {{27, 29}, {2, 3}},
          {{23, 30}, {20, 22}}, {{24, 31}, {2, 3}},   {{21, 32}, {2, 3}},
          {{25, 33}, {20, 21}}, {{20, 34}, {2, 3}},   {{1, 35}, {1, 2}},
          {{0, 36}, {0, 1}},    {{9, 37}, {3, 5}},    {{22, 38}, {15, 21}},
          {{23, 39}, {15, 20}}, {{15, 40}, {3, 14}},  {{22, 41}, {0, 15}},
          {{7, 42}, {3, 5}},    {{13, 43}, {0, 3}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 26u};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{0, 36}, {26, 0, 1, 0, 1}},
          {{1, 35}, {27, 1, 2, 1, 2}},
          {{1, 36}, {28, 26, 27, 0, 1}},
          {{2, 22}, {29, 2, 4, 2, 4}},
          {{2, 35}, {30, 27, 29, 1, 2}},
          {{2, 36}, {31, 28, 30, 0, 1}},
          {{3, 21}, {32, 4, 6, 4, 6}},
          {{3, 22}, {33, 29, 32, 2, 4}},
          {{3, 35}, {34, 30, 33, 1, 2}},
          {{3, 36}, {35, 31, 34, 0, 1}},
          {{4, 28}, {36, 5, 7, 5, 7}},
          {{5, 18}, {37, 7, 8, 7, 8}},
          {{5, 28}, {38, 36, 37, 5, 7}},
          {{6, 15}, {39, 8, 9, 8, 9}},
          {{6, 18}, {40, 37, 39, 7, 8}},
          {{6, 28}, {41, 38, 40, 5, 7}},
          {{7, 42}, {42, 3, 41, 3, 5}},
          {{8, 12}, {43, 9, 10, 9, 10}},
          {{8, 15}, {44, 39, 43, 8, 9}},
          {{8, 18}, {45, 40, 44, 7, 8}},
          {{8, 28}, {46, 41, 45, 5, 7}},
          {{8, 42}, {47, 42, 46, 3, 5}},
          {{9, 37}, {48, 3, 46, 3, 5}},
          {{10, 14}, {49, 6, 11, 6, 11}},
          {{10, 21}, {50, 32, 49, 4, 6}},
          {{10, 22}, {51, 33, 50, 2, 4}},
          {{10, 35}, {52, 34, 51, 1, 2}},
          {{10, 36}, {53, 35, 52, 0, 1}},
          {{11, 6}, {54, 11, 12, 11, 12}},
          {{11, 14}, {55, 49, 54, 6, 11}},
          {{11, 21}, {56, 50, 55, 4, 6}},
          {{11, 22}, {57, 51, 56, 2, 4}},
          {{11, 35}, {58, 52, 57, 1, 2}},
          {{11, 36}, {59, 53, 58, 0, 1}},
          {{12, 25}, {60, 3, 45, 3, 7}},
          {{12, 28}, {61, 60, 46, 3, 5}},
          {{13, 43}, {62, 59, 61, 0, 3}},
          {{14, 7}, {63, 10, 13, 10, 13}},
          {{14, 12}, {64, 43, 63, 9, 10}},
          {{14, 15}, {65, 44, 64, 8, 9}},
          {{14, 18}, {66, 45, 65, 7, 8}},
          {{14, 25}, {67, 60, 66, 3, 7}},
          {{14, 28}, {68, 61, 67, 3, 3}},
          {{14, 43}, {69, 62, 68, 0, 3}},
          {{15, 40}, {70, 68, 14, 3, 14}},
          {{15, 43}, {71, 69, 70, 0, 3}},
          {{16, 5}, {72, 13, 16, 13, 16}},
          {{16, 7}, {73, 63, 72, 10, 13}},
          {{16, 12}, {74, 64, 73, 9, 10}},
          {{16, 15}, {75, 65, 74, 8, 9}},
          {{16, 18}, {76, 66, 75, 7, 8}},
          {{16, 25}, {77, 67, 76, 3, 7}},
          {{16, 28}, {78, 68, 77, 3, 3}},
          {{16, 40}, {79, 70, 78, 3, 3}},
          {{16, 43}, {80, 71, 79, 0, 3}},
          {{17, 3}, {81, 16, 17, 16, 17}},
          {{17, 5}, {82, 72, 81, 13, 16}},
          {{17, 7}, {83, 73, 82, 10, 13}},
          {{17, 12}, {84, 74, 83, 9, 10}},
          {{17, 15}, {85, 75, 84, 8, 9}},
          {{17, 18}, {86, 76, 85, 7, 8}},
          {{17, 25}, {87, 77, 86, 3, 7}},
          {{17, 28}, {88, 78, 87, 3, 3}},
          {{17, 40}, {89, 79, 88, 3, 3}},
          {{17, 43}, {90, 80, 89, 0, 3}},
          {{18, 2}, {91, 17, 18, 17, 18}},
          {{18, 3}, {92, 81, 91, 16, 17}},
          {{18, 5}, {93, 82, 92, 13, 16}},
          {{18, 7}, {94, 83, 93, 10, 13}},
          {{18, 12}, {95, 84, 94, 9, 10}},
          {{18, 15}, {96, 85, 95, 8, 9}},
          {{18, 18}, {97, 86, 96, 7, 8}},
          {{18, 25}, {98, 87, 97, 3, 7}},
          {{18, 28}, {99, 88, 98, 3, 3}},
          {{18, 40}, {100, 89, 99, 3, 3}},
          {{18, 43}, {101, 90, 100, 0, 3}},
          {{19, 1}, {102, 18, 19, 18, 19}},
          {{19, 2}, {103, 91, 102, 17, 18}},
          {{19, 3}, {104, 92, 103, 16, 17}},
          {{19, 5}, {105, 93, 104, 13, 16}},
          {{19, 7}, {106, 94, 105, 10, 13}},
          {{19, 12}, {107, 95, 106, 9, 10}},
          {{19, 15}, {108, 96, 107, 8, 9}},
          {{19, 18}, {109, 97, 108, 7, 8}},
          {{19, 25}, {110, 98, 109, 3, 7}},
          {{19, 28}, {111, 99, 110, 3, 3}},
          {{19, 40}, {112, 100, 111, 3, 3}},
          {{19, 43}, {113, 101, 112, 0, 3}},
          {{20, 34}, {114, 57, 111, 2, 3}},
          {{20, 35}, {115, 58, 114, 1, 2}},
          {{20, 36}, {116, 59, 115, 0, 1}},
          {{20, 40}, {117, 116, 112, 0, 3}},
          {{21, 32}, {118, 57, 111, 2, 3}},
          {{22, 38}, {119, 15, 21, 15, 21}},
          {{22, 41}, {120, 117, 119, 0, 15}},
          {{23, 30}, {121, 20, 22, 20, 22}},
          {{23, 39}, {122, 119, 121, 15, 20}},
          {{23, 41}, {123, 120, 122, 0, 15}},
          {{24, 31}, {124, 57, 111, 2, 3}},
          {{25, 19}, {125, 21, 23, 21, 23}},
          {{25, 33}, {126, 121, 125, 20, 21}},
          {{25, 38}, {127, 119, 126, 15, 20}},
          {{25, 41}, {128, 123, 127, 0, 15}},
          {{26, 11}, {129, 22, 24, 22, 24}},
          {{26, 17}, {130, 129, 23, 22, 23}},
          {{26, 19}, {131, 125, 130, 21, 22}},
          {{26, 30}, {132, 121, 131, 20, 21}},
          {{26, 38}, {133, 127, 132, 15, 20}},
          {{26, 41}, {134, 128, 133, 0, 15}},
          {{27, 29}, {135, 57, 111, 2, 3}},
          {{28, 27}, {136, 110, 131, 3, 21}},
          {{28, 28}, {137, 111, 136, 3, 3}},
          {{28, 29}, {138, 135, 137, 2, 3}},
          {{28, 30}, {139, 138, 132, 2, 20}},
          {{28, 35}, {140, 115, 139, 1, 2}},
          {{28, 36}, {141, 116, 140, 0, 1}},
          {{28, 38}, {142, 141, 133, 0, 15}},
          {{28, 40}, {143, 117, 142, 0, 0}},
          {{29, 9}, {144, 24, 25, 24, 25}},
          {{29, 11}, {145, 129, 144, 22, 24}},
          {{29, 13}, {146, 145, 23, 22, 23}},
          {{29, 19}, {147, 131, 146, 21, 22}},
          {{29, 27}, {148, 136, 147, 3, 21}},
          {{29, 28}, {149, 137, 148, 3, 3}},
          {{29, 29}, {150, 138, 149, 2, 3}},
          {{29, 30}, {151, 139, 150, 2, 2}},
          {{29, 35}, {152, 140, 151, 1, 2}},
          {{29, 36}, {153, 141, 152, 0, 1}},
          {{29, 38}, {154, 142, 153, 0, 0}},
          {{29, 40}, {155, 143, 154, 0, 0}},
          {{30, 10}, {156, 23, 144, 23, 24}},
          {{30, 11}, {157, 145, 156, 22, 23}},
          {{31, 20}, {158, 109, 147, 7, 21}},
          {{31, 25}, {159, 110, 158, 3, 7}},
          {{32, 8}, {160, 106, 25, 10, 25}},
          {{32, 9}, {161, 160, 144, 10, 24}},
          {{32, 10}, {162, 161, 156, 10, 23}},
          {{32, 11}, {163, 162, 157, 10, 22}},
          {{32, 12}, {164, 107, 163, 9, 10}},
          {{32, 15}, {165, 108, 164, 8, 9}},
          {{32, 18}, {166, 109, 165, 7, 8}},
          {{32, 19}, {167, 166, 147, 7, 21}},
          {{33, 0}, {168, 19, 25, 19, 25}},
          {{33, 1}, {169, 102, 168, 18, 19}},
          {{33, 2}, {170, 103, 169, 17, 18}},
          {{33, 3}, {171, 104, 170, 16, 17}},
          {{33, 5}, {172, 105, 171, 13, 16}},
          {{33, 7}, {173, 106, 172, 10, 13}},
          {{34, 26}, {174, 57, 159, 2, 3}},
          {{34, 28}, {175, 174, 149, 2, 3}},
          {{35, 24}, {176, 57, 167, 2, 7}},
          {{35, 25}, {177, 176, 159, 2, 3}},
          {{36, 23}, {178, 57, 167, 2, 7}},
          {{37, 16}, {179, 55, 165, 6, 8}},
          {{37, 18}, {180, 179, 166, 6, 7}},
          {{37, 19}, {181, 180, 167, 6, 7}},
          {{37, 21}, {182, 56, 181, 4, 6}},
          {{37, 22}, {183, 57, 182, 2, 4}},
          {{38, 4}, {184, 12, 171, 12, 16}},
          {{38, 5}, {185, 184, 172, 12, 13}},
          {{38, 6}, {186, 54, 185, 11, 12}},
          {{38, 7}, {187, 173, 186, 10, 11}},
          {{38, 9}, {188, 161, 187, 10, 10}},
          {{38, 10}, {189, 162, 188, 10, 10}},
          {{38, 11}, {190, 163, 189, 10, 10}},
          {{38, 12}, {191, 164, 190, 9, 10}},
          {{38, 14}, {192, 55, 191, 6, 9}},
          {{38, 15}, {193, 192, 165, 6, 8}},
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("normal child") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{0, 2}, {0, 1}}, {{1, 0}, {1, 2}}, {{1, 1}, {0, 1}}
      }
  };
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{0, 2}, {3, 0, 1, 0, 1}},
          {{1, 0}, {4, 1, 2, 1, 2}},
          {{1, 1}, {5, 0, 4, 0, 1}}
      }
  };
  linkage_hierarchy_t actual{minpres.grades(), minpres.edges(), 3u};
  CHECK(actual == expected);
}

TEST_CASE("conflict child") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 2}, {0, 1}}, {{2, 0}, {1, 2}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 3};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 2}, {3, 0, 1, 0, 1}},
          {{2, 0}, {4, 1, 2, 1, 2}},
          {{2, 2}, {5, 3, 4, 0, 1}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("mixed child") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 4}, {0, 1}},
          {{2, 3}, {1, 2}},
          {{3, 0}, {2, 3}},
          {{3, 1}, {1, 2}},
          {{3, 2}, {0, 1}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 4};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 4}, {4, 0, 1, 0, 1}},
          {{2, 3}, {5, 1, 2, 1, 2}},
          {{2, 4}, {6, 4, 5, 0, 1}},
          {{3, 0}, {7, 2, 3, 2, 3}},
          {{3, 1}, {8, 1, 7, 1, 2}},
          {{3, 2}, {9, 0, 8, 0, 1}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("split child") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 2}, {0, 1}}, {{2, 3}, {0, 2}}, {{3, 0}, {2, 3}}, {{3, 1}, {1, 2}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 4};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 2}, {4, 0, 1, 0, 1}},
          {{2, 3}, {5, 4, 2, 0, 2}},
          {{3, 0}, {6, 2, 3, 2, 3}},
          {{3, 1}, {7, 1, 6, 1, 2}},
          {{3, 2}, {8, 4, 7, 0, 1}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("normal sibling") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 0}, {0, 1}}, {{2, 1}, {0, 2}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 3};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 0}, {3, 0, 1, 0, 1}}, {{2, 1}, {4, 3, 2, 0, 2}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("conflict sibling") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 1}, {0, 1}}, {{2, 0}, {0, 2}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 3};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 1}, {3, 0, 1, 0, 1}},
          {{2, 0}, {4, 0, 2, 0, 2}},
          {{2, 1}, {5, 3, 4, 0, 0}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("conflict child and normal sibling") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 1}, {0, 1}}, {{2, 0}, {1, 2}}, {{3, 2}, {0, 3}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 4};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 1}, {4, 0, 1, 0, 1}},
          {{2, 0}, {5, 1, 2, 1, 2}},
          {{2, 1}, {6, 4, 5, 0, 1}},
          {{3, 2}, {7, 6, 3, 0, 3}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("conflict child and full conflict sibling") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 2}, {0, 1}}, {{2, 1}, {1, 2}}, {{3, 0}, {0, 3}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 4};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 2}, {4, 0, 1, 0, 1}},
          {{2, 1}, {5, 1, 2, 1, 2}},
          {{2, 2}, {6, 4, 5, 0, 1}},
          {{3, 0}, {7, 0, 3, 0, 3}},
          {{3, 2}, {8, 6, 7, 0, 0}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("conflict child and partial conflict sibling (a)") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 1}, {0, 1}}, {{2, 2}, {0, 2}}, {{3, 0}, {1, 3}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 4};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 1}, {4, 0, 1, 0, 1}},
          {{2, 2}, {5, 4, 2, 0, 2}},
          {{3, 0}, {6, 1, 3, 1, 3}},
          {{3, 1}, {7, 4, 6, 0, 1}},
          {{3, 2}, {8, 5, 7, 0, 0}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("conflict child and partial conflict sibling (b)") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 2}, {0, 1}}, {{2, 0}, {0, 2}}, {{3, 1}, {1, 3}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 4};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 2}, {4, 0, 1, 0, 1}},
          {{2, 0}, {5, 0, 2, 0, 2}},
          {{2, 2}, {6, 4, 5, 0, 0}},
          {{3, 1}, {7, 1, 3, 1, 3}},
          {{3, 2}, {8, 6, 7, 0, 1}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("two conflict siblings") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 2}, {0, 1}}, {{2, 1}, {0, 2}}, {{3, 0}, {0, 3}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 4};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 2}, {4, 0, 1, 0, 1}},
          {{2, 1}, {5, 0, 2, 0, 2}},
          {{2, 2}, {6, 4, 5, 0, 0}},
          {{3, 0}, {7, 0, 3, 0, 3}},
          {{3, 1}, {8, 5, 7, 0, 0}},
          {{3, 2}, {9, 6, 8, 0, 0}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("normal siblings with conflict children") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 1}, {0, 1}}, {{2, 0}, {1, 2}}, {{3, 3}, {0, 3}}, {{4, 2}, {3, 4}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 5};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 1}, {5, 0, 1, 0, 1}},
          {{2, 0}, {6, 1, 2, 1, 2}},
          {{2, 1}, {7, 5, 6, 0, 1}},
          {{3, 3}, {8, 7, 3, 0, 3}},
          {{4, 2}, {9, 3, 4, 3, 4}},
          {{4, 3}, {10, 8, 9, 0, 3}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("ancestor siblings") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 4}, {0, 1}},
          {{2, 3}, {1, 2}},
          {{3, 0}, {1, 3}},
          {{3, 2}, {0, 1}},
          {{4, 1}, {1, 4}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 5};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 4}, {5, 0, 1, 0, 1}},
          {{2, 3}, {6, 1, 2, 1, 2}},
          {{2, 4}, {7, 5, 6, 0, 1}},
          {{3, 0}, {8, 1, 3, 1, 3}},
          {{3, 2}, {9, 0, 8, 0, 1}},
          {{3, 3}, {10, 9, 6, 0, 1}},
          {{4, 1}, {11, 8, 4, 1, 4}},
          {{4, 2}, {12, 9, 11, 0, 1}},
          {{4, 3}, {13, 10, 12, 0, 0}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("multilevel ancestors") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 9}, {0, 1}},
          {{2, 8}, {1, 2}},
          {{3, 7}, {2, 3}},
          {{4, 6}, {3, 4}},
          {{5, 4}, {3, 5}},
          {{5, 5}, {2, 3}},
          {{6, 3}, {3, 6}},
          {{7, 1}, {1, 7}},
          {{7, 2}, {0, 1}},
          {{8, 0}, {1, 8}}
      }
  };
  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 9};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 9}, {9, 0, 1, 0, 1}},    {{2, 8}, {10, 1, 2, 1, 2}},
          {{2, 9}, {11, 9, 10, 0, 1}},  {{3, 7}, {12, 2, 3, 2, 3}},
          {{3, 8}, {13, 10, 12, 1, 2}}, {{3, 9}, {14, 11, 13, 0, 1}},
          {{4, 6}, {15, 3, 4, 3, 4}},   {{4, 7}, {16, 12, 15, 2, 3}},
          {{4, 8}, {17, 13, 16, 1, 2}}, {{4, 9}, {18, 14, 17, 0, 1}},
          {{5, 4}, {19, 3, 5, 3, 5}},   {{5, 5}, {20, 2, 19, 2, 3}},
          {{5, 6}, {21, 20, 15, 2, 3}}, {{5, 8}, {22, 17, 21, 1, 2}},
          {{5, 9}, {23, 18, 22, 0, 1}}, {{6, 3}, {24, 3, 6, 3, 6}},
          {{6, 4}, {25, 19, 24, 3, 3}}, {{6, 5}, {26, 20, 25, 2, 3}},
          {{6, 6}, {27, 21, 26, 2, 2}}, {{6, 8}, {28, 22, 27, 1, 2}},
          {{6, 9}, {29, 23, 28, 0, 1}}, {{7, 1}, {30, 1, 7, 1, 7}},
          {{7, 2}, {31, 0, 30, 0, 1}},  {{7, 8}, {32, 31, 28, 0, 1}},
          {{8, 0}, {33, 1, 8, 1, 8}},   {{8, 1}, {34, 30, 33, 1, 1}},
          {{8, 2}, {35, 31, 34, 0, 1}}, {{8, 8}, {36, 32, 35, 0, 0}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("passthrough edge and ancestor sibling") {
  minimal_presentation_t const minpres{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, edge_t<unsigned>>>{
          {{1, 11}, {0, 1}},
          {{2, 8}, {0, 2}},
          {{2, 10}, {0, 1}},
          {{3, 9}, {1, 3}},
          {{4, 5}, {2, 4}},
          {{5, 0}, {4, 5}},
          {{6, 3}, {2, 6}},
          {{6, 6}, {1, 2}},
          {{6, 7}, {0, 1}},
          {{7, 4}, {4, 7}},
          {{8, 1}, {7, 8}},
          {{8, 2}, {4, 7}}
      }
  };

  linkage_hierarchy_t const actual{minpres.grades(), minpres.edges(), 9};
  linkage_hierarchy_t const expected{
      std::from_range,
      std::vector<std::pair<bigrade_t<unsigned>, link_t<unsigned>>>{
          {{1, 11}, {9, 0, 1, 0, 1}},    {{2, 8}, {10, 0, 2, 0, 2}},
          {{2, 10}, {11, 10, 1, 0, 1}},  {{3, 9}, {12, 1, 3, 1, 3}},
          {{3, 10}, {13, 11, 12, 0, 1}}, {{4, 5}, {14, 2, 4, 2, 4}},
          {{4, 8}, {15, 10, 14, 0, 2}},  {{4, 10}, {16, 13, 15, 0, 0}},
          {{5, 0}, {17, 4, 5, 4, 5}},    {{5, 5}, {18, 14, 17, 2, 4}},
          {{5, 8}, {19, 15, 18, 0, 2}},  {{5, 10}, {20, 16, 19, 0, 0}},
          {{6, 3}, {21, 2, 6, 2, 6}},    {{6, 5}, {22, 18, 21, 2, 2}},
          {{6, 6}, {23, 1, 22, 1, 2}},   {{6, 7}, {24, 0, 23, 0, 1}},
          {{6, 9}, {25, 24, 12, 0, 1}},  {{7, 4}, {26, 17, 7, 4, 7}},
          {{7, 5}, {27, 22, 26, 2, 4}},  {{7, 6}, {28, 23, 27, 1, 2}},
          {{7, 7}, {29, 24, 28, 0, 1}},  {{7, 9}, {30, 25, 29, 0, 0}},
          {{8, 1}, {31, 7, 8, 7, 8}},    {{8, 2}, {32, 17, 31, 4, 7}},
          {{8, 5}, {33, 27, 32, 2, 4}},  {{8, 6}, {34, 28, 33, 1, 2}},
          {{8, 7}, {35, 29, 34, 0, 1}},  {{8, 9}, {36, 30, 35, 0, 0}}
      }
  };
  CHECK(actual == expected);
}

TEST_CASE("larger example (horse)") {
  // Configure input
  std::vector<double> const distances =
      npy::load<double>("../../tests/data/horse_distance.npy");
  std::vector<double> const lens_values =
      npy::load<double>("../../tests/data/horse_lens.npy");
  biperscan_minpres_result_t res = biperscan_minpres<unsigned>(
      distances, lens_values
  );
  linkage_hierarchy_t hierarchy = biperscan_linkage(
      res.minimal_presentation, lens_values.size()
  );

  // Test output
  CHECK(hierarchy.size() < distances.size());
  CHECK(hierarchy.size() >= res.minimal_presentation.size());
  CHECK(hierarchy.links()[0].id == lens_values.size());

  // Compute link members, check all links add novel points.
  std::size_t const num_points = lens_values.size();
  std::size_t const linkage_size = hierarchy.size();
  std::vector<std::vector<unsigned>> members{linkage_size};
  for (auto const &[i, l] : std::views::enumerate(hierarchy.links())) {
    // Use pointers to describe member ranges (avoiding explicit singles)
    unsigned const *parent_begin, *child_begin, *parent_end, *child_end;
    if (l.parent < num_points) {
      parent_begin = &l.parent;
      parent_end = parent_begin + 1u;
    } else {
      parent_begin = &members[l.parent - num_points][0];
      parent_end = parent_begin + members[l.parent - num_points].size();
    }
    if (l.child < num_points) {
      child_begin = &l.child;
      child_end = child_begin + 1u;
    } else {
      child_begin = &members[l.child - num_points][0];
      child_end = child_begin + members[l.child - num_points].size();
    }

    std::set_union(
        parent_begin, parent_end, child_begin, child_end,
        std::back_inserter(members[i])
    );

    REQUIRE(
        static_cast<std::ptrdiff_t>(members[i].size()) >
        (parent_end - parent_begin)
    );
    REQUIRE(
        static_cast<std::ptrdiff_t>(members[i].size()) >
        (child_end - child_begin)
    );
  }
}

TEST_SUITE_END();