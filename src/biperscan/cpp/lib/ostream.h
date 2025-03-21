#ifndef BIPERSCAN_LIB_OSTREAM_H
#define BIPERSCAN_LIB_OSTREAM_H

#include <iomanip>
#include <ostream>

#include "base_types.h"

namespace bppc {

namespace detail {
extern std::size_t WIDTH;   // # digits for grades, point indices
extern std::size_t HWIDTH;  // # digits for link ids, parents, children
void set_width(std::size_t max_value);
void set_hwidth(std::size_t max_value);

// Concepts for printing graded matrices.
template <typename type_t>
concept printable_bigrade = requires(type_t const &g) {
  requires std::unsigned_integral<std::decay_t<decltype(g.lens)>>;
  requires std::unsigned_integral<std::decay_t<decltype(g.distance)>>;
  operator<<(std::declval<std::ostream &>(), g);
};

template <typename type_t>
concept printable_edge = requires(type_t const &e) {
  requires std::unsigned_integral<std::decay_t<decltype(e.parent)>>;
  requires std::unsigned_integral<std::decay_t<decltype(e.child)>>;
  operator<<(std::declval<std::ostream &>(), e);
};

template <typename type_t>
concept printable_link = requires(type_t const &l) {
  requires printable_edge<type_t>;
  requires std::unsigned_integral<std::decay_t<decltype(l.id)>>;
  requires std::unsigned_integral<std::decay_t<decltype(l.parent_root)>>;
  requires std::unsigned_integral<std::decay_t<decltype(l.child_root)>>;
  operator<<(std::declval<std::ostream &>(), l);
};

template <typename concept_t>
concept printable_matrix = requires(concept_t const &m) {
  { m.size() } -> std::convertible_to<std::size_t>;
  { m.grades() } -> std::ranges::forward_range;
  { m.items() } -> std::ranges::forward_range;
  { *m.grades().begin() } -> printable_bigrade;
  { *std::views::keys(m.items()).begin() } -> printable_bigrade;
  { *std::views::values(m.items()).begin() } -> printable_edge;
};

template <typename concept_t>
concept printable_linkage_matrix = requires(concept_t const &m) {
  requires printable_matrix<concept_t>;
  { m.links() } -> std::ranges::forward_range;
  { *m.links().begin() } -> printable_link;
  { *std::views::values(m.items()).begin() } -> printable_link;
};

}  // namespace detail

template <std::unsigned_integral grade_t>
std::ostream &operator<<(std::ostream &os, bigrade_t<grade_t> const &g) {
  return os << '{' << std::setw(detail::WIDTH) << g.lens << ", "
            << std::setw(detail::WIDTH) << g.distance << '}';
}

template <std::unsigned_integral index_t>
std::ostream &operator<<(std::ostream &os, edge_t<index_t> const &e) {
  return os << std::setw(detail::WIDTH) << e.parent << "<-"
            << std::setw(detail::WIDTH) << std::left << e.child << std::right;
}

template <std::unsigned_integral index_t>
std::ostream &operator<<(std::ostream &os, link_t<index_t> const &e) {
  return os << std::setw(detail::HWIDTH) << e.id << ": "
            << std::setw(detail::HWIDTH) << e.parent << "<-"
            << std::setw(detail::HWIDTH) << std::left << e.child << std::right
            << " (" << edge_t{e.parent_root, e.child_root} << ')';
}

template <std::unsigned_integral index_t>
std::ostream &operator<<(std::ostream &os, merge_t<index_t> const &e) {
  return os << std::setw(detail::HWIDTH) << e.start_column << ", "
            << std::setw(detail::HWIDTH) << e.end_column << ", "
            << std::setw(detail::HWIDTH) << e.parent << ", "
            << std::setw(detail::HWIDTH) << e.child << ", "
            << std::setw(detail::HWIDTH) << e.parent_side.size() << ", "
            << std::setw(detail::HWIDTH) << e.child_side.size();
}

template <detail::printable_matrix matrix_t>
std::ostream &operator<<(std::ostream &os, matrix_t &&m) {
  std::size_t const max_point = std::ranges::max(std::views::transform(
      m.grades(), [](auto const &c) { return std::max(c.lens, c.distance); }
  ));
  detail::set_width(max_point);
  detail::set_hwidth(max_point + m.size());
  for (auto &&[grade, col] : m.items())
    os << grade << ' ' << col << '\n';
  detail::WIDTH = 1u;
  detail::HWIDTH = 1u;
  return os << '\n';
}

template <detail::printable_matrix matrix_t>
std::ostream &print_table(std::ostream &os, matrix_t &&m) {
  // Extract grades and sort them
  // Configure number width and column width
  std::size_t const num_points = std::ranges::max(
      std::views::transform(m.edges(), [](auto const &c) { return c.child; })
  );
  std::size_t const max_point = std::ranges::max(std::views::transform(
      m.grades(), [](auto const &c) { return std::max(c.lens, c.distance); }
  ));
  if constexpr (detail::printable_linkage_matrix<matrix_t>)
    detail::set_width(max_point + m.size());
  else
    detail::set_width(max_point);

  // print grades row
  os << std::setw(detail::WIDTH) << ' ' << "; ";
  for (auto &&g : m.grades())
    os << std::setw(detail::WIDTH) << g.lens << ", " << std::setw(detail::WIDTH)
       << g.distance << " ; ";
  os << '\n';

  // Print edge row
  os << std::setw(detail::WIDTH) << ' ' << "; ";
  for (auto &&e : std::views::values(m.items()))
    os << std::setw(detail::WIDTH) << e.parent << "<-"
       << std::setw(detail::WIDTH) << std::left << e.child << std::right
       << " ; ";
  os << '\n';

  // Print matrix rows
  for (unsigned point : std::views::iota(0u, num_points + 1)) {
    os << std::setw(detail::WIDTH) << point << "; ";
    for (auto &&e : std::views::values(m.items()))
      os << std::setw(detail::WIDTH + 1)
         << (e.parent == point or e.child == point ? '1' : ' ')
         << std::setw(detail::WIDTH + 1) << ' ' << " ; ";
    os << '\n';
  }
  detail::WIDTH = 1u;
  return os;
}

}  // namespace bppc

#endif  // BIPERSCAN_LIB_OSTREAM_H
