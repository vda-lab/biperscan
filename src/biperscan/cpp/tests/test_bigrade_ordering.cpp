#include "doctest.h"
#include "lib/bigrade_ordering.h"

using namespace bppc;

TEST_SUITE_BEGIN("bigrade ordering");

TEST_CASE("comparison") {
  SUBCASE("equality") {
    CHECK(bigrade_t<unsigned>{} == bigrade_t<unsigned>{});
    CHECK(bigrade_t<unsigned>{} != bigrade_t<unsigned>{1, 1});
  }

  SUBCASE("equality") {
    CHECK(bigrade_t<unsigned>{} == bigrade_t<unsigned>{});
    CHECK(bigrade_t<unsigned>{} != bigrade_t<unsigned>{1, 1});
  }

  SUBCASE("bigrade_t less") {
    constexpr bigrade_less<unsigned> comp{};
    CHECK(comp({0, 0}, {1, 1}));
    CHECK(not comp({0, 1}, {1, 1}));
    CHECK(not comp({1, 0}, {1, 1}));
    CHECK(not comp({1, 1}, {1, 1}));
    CHECK(not comp({2, 0}, {1, 1}));
    CHECK(not comp({0, 2}, {1, 1}));
    CHECK(not comp({2, 2}, {1, 1}));
  }

  SUBCASE("bigrade_t greater") {
    constexpr bigrade_greater<unsigned> comp{};
    CHECK(not comp({0, 0}, {1, 1}));
    CHECK(not comp({0, 1}, {1, 1}));
    CHECK(not comp({1, 0}, {1, 1}));
    CHECK(not comp({1, 1}, {1, 1}));
    CHECK(not comp({2, 0}, {1, 1}));
    CHECK(not comp({0, 2}, {1, 1}));
    CHECK(comp({2, 2}, {1, 1}));
  }

  SUBCASE("bigrade_t lex less") {
    constexpr bigrade_lex_less<unsigned> comp{};
    CHECK(comp({0, 0}, {1, 1}));
    CHECK(comp({0, 1}, {1, 1}));
    CHECK(comp({1, 0}, {1, 1}));
    CHECK(not comp({1, 1}, {1, 1}));
    CHECK(not comp({2, 0}, {1, 1}));
    CHECK(comp({0, 2}, {1, 1}));
    CHECK(not comp({2, 1}, {1, 1}));
    CHECK(not comp({1, 2}, {1, 1}));
    CHECK(not comp({2, 2}, {1, 1}));
  }

  SUBCASE("bigrade_t lex greater") {
    constexpr bigrade_lex_greater<unsigned> comp{};
    CHECK(not comp({0, 0}, {1, 1}));
    CHECK(not comp({0, 1}, {1, 1}));
    CHECK(not comp({1, 0}, {1, 1}));
    CHECK(not comp({1, 1}, {1, 1}));
    CHECK(comp({2, 0}, {1, 1}));
    CHECK(not comp({0, 2}, {1, 1}));
    CHECK(comp({2, 1}, {1, 1}));
    CHECK(comp({1, 2}, {1, 1}));
    CHECK(comp({2, 2}, {1, 1}));
  }

  SUBCASE("bigrade_t colex less") {
    constexpr bigrade_colex_less<unsigned> comp{};
    CHECK(comp({0, 0}, {1, 1}));
    CHECK(comp({0, 1}, {1, 1}));
    CHECK(comp({1, 0}, {1, 1}));
    CHECK(not comp({1, 1}, {1, 1}));
    CHECK(comp({2, 0}, {1, 1}));
    CHECK(not comp({0, 2}, {1, 1}));
    CHECK(not comp({2, 1}, {1, 1}));
    CHECK(not comp({1, 2}, {1, 1}));
    CHECK(not comp({2, 2}, {1, 1}));
  }

  SUBCASE("bigrade_t colex greater") {
    constexpr bigrade_colex_greater<unsigned> comp{};
    CHECK(not comp({0, 0}, {1, 1}));
    CHECK(not comp({0, 1}, {1, 1}));
    CHECK(not comp({1, 0}, {1, 1}));
    CHECK(not comp({1, 1}, {1, 1}));
    CHECK(not comp({2, 0}, {1, 1}));
    CHECK(comp({0, 2}, {1, 1}));
    CHECK(comp({2, 1}, {1, 1}));
    CHECK(comp({1, 2}, {1, 1}));
    CHECK(comp({2, 2}, {1, 1}));
  }
}

TEST_SUITE_END();
