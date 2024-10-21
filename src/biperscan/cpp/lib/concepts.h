#ifndef BIPERSCAN_LIB_RANGES_H
#define BIPERSCAN_LIB_RANGES_H

#include <ranges>

#include "base_types.h"

namespace bppc {

// Deduction guides
namespace detail {
template <typename>
struct template_type : std::false_type {};

template <
    std::unsigned_integral templated_t,
    template <std::unsigned_integral> typename template_t>
struct template_type<template_t<templated_t>> : std::true_type {
  using type = templated_t;
};

template <
    std::unsigned_integral templated_t,
    template <std::unsigned_integral> typename template_t>
struct template_type<template_t<templated_t const>> : std::true_type {
  using type = templated_t;
};

template <typename range_t>
concept graded_range =
    std::ranges::input_range<range_t> and
    std::ranges::forward_range<range_t> and
    template_type<std::ranges::range_value_t<
        decltype(std::views::keys(std::declval<range_t>()))>>::value and
    template_type<std::ranges::range_value_t<
        decltype(std::views::values(std::declval<range_t>()))>>::value;
}  // namespace detail

template <typename range_t, typename value_t>
concept range_of =
    std::ranges::viewable_range<range_t> and
    std::ranges::input_range<range_t> and
    std::ranges::forward_range<range_t> and
    std::convertible_to<std::ranges::range_reference_t<range_t>, value_t>;

template <typename range_t, typename value_t>
concept view_of =
    std::ranges::view<range_t> and std::ranges::input_range<range_t> and
    std::ranges::forward_range<range_t> and
    std::convertible_to<std::ranges::range_reference_t<range_t>, value_t>;

template <
    typename view_t, template <std::unsigned_integral> typename template_t>
concept templated_range_of = range_of<
    view_t, template_t<typename detail::template_type<
                std::ranges::range_value_t<view_t>>::type>>;

template <
    typename view_t, template <std::unsigned_integral> typename template_t>
concept templated_view_of = view_of<
    view_t, template_t<typename detail::template_type<
                std::ranges::range_value_t<view_t>>::type>>;

};  // namespace bppc

#endif  // BIPERSCAN_LIB_RANGES_H
