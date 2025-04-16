#ifndef BIPERSCAN_LIB_LINKAGE_HIERARCHY_H
#define BIPERSCAN_LIB_LINKAGE_HIERARCHY_H

#include "minimal_presentation.h"

namespace bppc {

/**
 * @brief A type that stores a bigraded linkage hierarchy presentation.
 * For use, first create the hierarchy object, which will reserve space for an
 * expected number of links. Then, fill the hierarchy by calling `add_link`.
 * @tparam index_t - Precision for storing unsigned indices.
 * @tparam grade_t - Precision for storing unsigned grades.
 */
template <std::unsigned_integral index_t, std::unsigned_integral grade_t>
class linkage_hierarchy_t {
  // Type shorthands
  using edge_t = edge_t<index_t>;
  using link_t = link_t<index_t>;
  using bigrade_t = bigrade_t<grade_t>;
  using value_t = std::pair<bigrade_t, link_t>;
  using minimal_presentation_t = minimal_presentation_t<index_t, grade_t>;

  using pivot_map_t = std::map<grade_t, index_t, std::greater<>>;
  using pivot_it_t = typename pivot_map_t::iterator;
  using pivot_reverse_it_t = typename pivot_map_t::reverse_iterator;

#pragma pack(push, 1)
  struct delayed_merge_t {
    pivot_it_t parent_it{};
    index_t parent_root{};
    mutable index_t child_root{};
    // mutable index_t delaying_parent{};

    [[nodiscard]] bool operator==(delayed_merge_t const &) const = default;
    [[nodiscard]] auto operator<=>(delayed_merge_t const &other) const {
      return parent_it->first <=> other.parent_it->first;
    }
  };
#pragma pack(pop)

  using delay_queue_t = std::multiset<delayed_merge_t>;
  using queue_it_t = typename delay_queue_t::iterator;

  // Private data members
  index_t const d_num_points{};
  std::vector<grade_t> d_lens_grades;
  std::vector<grade_t> d_distance_grades;
  std::vector<index_t> d_parents;
  std::vector<index_t> d_children;
  std::vector<index_t> d_parent_roots;
  std::vector<index_t> d_child_roots;

 public:
  // Construct from known edges with sanity checks.
  template <std::ranges::sized_range link_range_t>
    requires range_of<link_range_t, value_t>
  linkage_hierarchy_t(std::from_range_t, link_range_t &&links)
      : d_num_points(links.empty() ? 0u : links[0].second.id),
        d_lens_grades(
            std::ranges::to<std::vector<grade_t>>(std::views::transform(
                links, [](auto const &link) { return link.first.lens; }
            ))
        ),
        d_distance_grades(
            std::ranges::to<std::vector<grade_t>>(std::views::transform(
                links, [](auto const &link) { return link.first.distance; }
            ))
        ),
        d_parents(std::ranges::to<std::vector<index_t>>(std::views::transform(
            links, [](auto const &link) { return link.second.parent; }
        ))),
        d_children(std::ranges::to<std::vector<index_t>>(std::views::transform(
            links, [](auto const &link) { return link.second.child; }
        ))),
        d_parent_roots(
            std::ranges::to<std::vector<index_t>>(std::views::transform(
                links, [](auto const &link) { return link.second.parent_root; }
            ))
        ),
        d_child_roots(
            std::ranges::to<std::vector<index_t>>(std::views::transform(
                links, [](auto const &link) { return link.second.child_root; }
            ))
        ) {
    // Sanity checks for expected values in testing!
    index_t id = d_num_points;
    for (auto const &link : std::views::values(links)) {
      if (id++ != link.id)
        throw std::runtime_error("Link ids must be in sequence!");
      if (link.child_root < link.parent_root)
        throw std::runtime_error(
            "Link parent root must be less than or equal to child root!"
        );
    }
    bigrade_lex_greater<grade_t> grade_comp{};
    for (auto const &[prev_grade, grade] : std::views::zip(
             std::views::keys(links),
             std::views::keys(links) | std::views::drop(1)
         )) {
      if (grade_comp(prev_grade, grade))
        throw std::runtime_error("Links must be in bigrade_lex_less order!");
    }
  }

  // Construct from a minpres.
  linkage_hierarchy_t(
      view_of<bigrade_t> auto const grades, view_of<edge_t> auto const edges,
      std::size_t const num_points
  )
      : d_num_points(static_cast<index_t>(num_points)) {
    // Initialise state
    reserve(grades.size());  // TODO: what is a good estimate?
    std::vector<pivot_map_t> pivots(num_points);  // per point
    delay_queue_t delay_queue{};                  // per lens grade

    // Iterate over minpres edges (bigrade_lex_less) (distances are unique)
    grade_t delay_lens{1u};  // different from actual first value
    for (auto const &[grade, edge] : std::views::zip(grades, edges)) {
      process_delayed_merges(pivots, delay_queue, grade, delay_lens);
      process_minpres_edge(pivots, delay_queue, grade, edge);
      delay_lens = grade.lens;
    }
    for (delayed_merge_t const &delayed_merge : delay_queue)
      process_delayed_merge(pivots, delay_lens, delayed_merge);

    // Release unused memory
    shrink_to_fit();
  }

  [[nodiscard]] bool operator==(linkage_hierarchy_t const &) const = default;

  // Individual column access by grade.
  [[nodiscard]] link_t link_at(bigrade_t const &grade) const {
    auto view = grades();
    auto const it = std::ranges::lower_bound(
        view, grade, bigrade_lex_less<grade_t>{}
    );
    auto const idx =
        static_cast<index_t>(std::ranges::distance(view.begin(), it));
    return link_t{
        idx + d_num_points, d_parents[idx], d_children[idx],
        d_parent_roots[idx], d_child_roots[idx]
    };
  }

  // Individual column access by link id (idx = id - num_points).
  [[nodiscard]] bool is_link(index_t id) const {
    return id >= d_num_points;
  }
  [[nodiscard]] bigrade_t grade_of(index_t id) const {
    return bigrade_t{
        .lens = lens_grade_of(id), .distance = distance_grade_of(id)
    };
  }
  [[nodiscard]] link_t link_of(index_t id) const {
    index_t idx = id - d_num_points;
    return link_t{
        id, d_parents[idx], d_children[idx], d_parent_roots[idx],
        d_child_roots[idx]
    };
  }
  [[nodiscard]] grade_t lens_grade_of(index_t id) const {
    return d_lens_grades[id - d_num_points];
  }
  [[nodiscard]] grade_t distance_grade_of(index_t id) const {
    return d_distance_grades[id - d_num_points];
  }
  [[nodiscard]] index_t parent_of(index_t id) const {
    return d_parents[id - d_num_points];
  }
  [[nodiscard]] index_t child_of(index_t id) const {
    return d_children[id - d_num_points];
  }
  [[nodiscard]] index_t parent_root_of(index_t id) const {
    return d_parent_roots[id - d_num_points];
  }
  [[nodiscard]] index_t child_root_of(index_t id) const {
    return d_child_roots[id - d_num_points];
  }

  // View of links in bigrade_colex_less order.
  [[nodiscard]] view_of<std::pair<bigrade_t, link_t>> auto items() const {
    return std::views::zip_transform(
        std::make_pair<bigrade_t, link_t>, grades(), links()
    );
  }
  [[nodiscard]] view_of<bigrade_t> auto grades() const {
    return std::views::zip_transform(
        [](auto... args) { return bigrade_t{args...}; },
        std::views::all(d_lens_grades), std::views::all(d_distance_grades)
    );
  }
  [[nodiscard]] view_of<link_t> auto links() const {
    return std::views::zip_transform(
        [](auto... args) { return link_t{args...}; },
        std::views::iota(d_num_points, d_num_points + d_parents.size()),
        std::views::all(d_parents), std::views::all(d_children),
        std::views::all(d_parent_roots), std::views::all(d_child_roots)
    );
  }

  // Memory management
  [[nodiscard]] bool empty() const {
    return d_parents.empty();
  }
  [[nodiscard]] std::size_t size() const {
    return d_parents.size();
  }
  void clear() {
    d_lens_grades.clear();
    d_distance_grades.clear();
    d_parents.clear();
    d_children.clear();
    d_parent_roots.clear();
    d_child_roots.clear();
  }

  // Python API takes ownership of the vectors to avoid copies.
  [[nodiscard]] std::vector<grade_t> &&take_lens_grades() {
    return std::move(d_lens_grades);
  }
  [[nodiscard]] std::vector<grade_t> &&take_distance_grades() {
    return std::move(d_distance_grades);
  }
  [[nodiscard]] std::vector<index_t> &&take_parents() {
    return std::move(d_parents);
  }
  [[nodiscard]] std::vector<index_t> &&take_children() {
    return std::move(d_children);
  }
  [[nodiscard]] std::vector<index_t> &&take_parent_roots() {
    return std::move(d_parent_roots);
  }
  [[nodiscard]] std::vector<index_t> &&take_child_roots() {
    return std::move(d_child_roots);
  }

 private:
  [[nodiscard]] index_t add_link(
      bigrade_t const &grade, index_t parent, index_t child,
      index_t parent_root, index_t child_root
  ) {
    auto const id = static_cast<index_t>(d_parents.size()) + d_num_points;
    d_lens_grades.push_back(grade.lens);
    d_distance_grades.push_back(grade.distance);
    d_parents.push_back(parent);
    d_children.push_back(child);
    d_parent_roots.push_back(parent_root);
    d_child_roots.push_back(child_root);
    return id;
  }
  void reserve(std::size_t const num_links) {
    d_lens_grades.reserve(num_links);
    d_distance_grades.reserve(num_links);
    d_parents.reserve(num_links);
    d_children.reserve(num_links);
    d_parent_roots.reserve(num_links);
    d_child_roots.reserve(num_links);
  }
  void shrink_to_fit() {
    d_lens_grades.shrink_to_fit();
    d_distance_grades.shrink_to_fit();
    d_parents.shrink_to_fit();
    d_children.shrink_to_fit();
    d_parent_roots.shrink_to_fit();
    d_child_roots.shrink_to_fit();
  }

  void process_delayed_merges(
      std::vector<pivot_map_t> &pivots, delay_queue_t &delay_queue,
      bigrade_t const &grade, grade_t const delay_lens
  ) {
    if (delay_lens != grade.lens) {
      for (delayed_merge_t const &delayed_merge : delay_queue)
        process_delayed_merge(pivots, delay_lens, delayed_merge);
      delay_queue.clear();
    } else {
      while (not delay_queue.empty() and
             delay_queue.begin()->parent_it->first < grade.distance) {
        queue_it_t delay_it = delay_queue.begin();
        process_delayed_merge(pivots, delay_lens, *delay_it);
        delay_queue.erase(delay_it);
      }
    }
  }
  void process_delayed_merge(
      std::vector<pivot_map_t> &pivots, grade_t const lens,
      delayed_merge_t merge
  ) {
    auto &[parent_it, parent_root, child_root] = merge;
    pivot_it_t child_it = pivots[child_root].upper_bound(parent_it->first);
    bigrade_t const grade{lens, parent_it->first};
    if (parent_root > child_root) {
      std::swap(parent_root, child_root);
      std::swap(parent_it, child_it);
    }

    index_t const parent = parent_it->second;
    index_t const child = child_it->second;
    index_t const id = add_link(grade, parent, child, parent_root, child_root);
    pivots[child_root].insert_or_assign(child_it, grade.distance, id);
    pivots[parent_root].insert_or_assign(parent_it, grade.distance, id);
  }
  void process_minpres_edge(
      std::vector<pivot_map_t> &pivots, delay_queue_t &delay_queue,
      bigrade_t const &grade, edge_t const &edge
  ) {
    // Find accessible pivots
    auto [prnt_it, prnt, prnt_root] = find_pivot(pivots, grade, edge.parent);
    auto [chld_it, chld, chld_root] = find_pivot(pivots, grade, edge.child);
    if (prnt == chld)
      return;
    if (chld_root < prnt_root) {
      std::swap(prnt_it, chld_it);
      std::swap(prnt, chld);
      std::swap(prnt_root, chld_root);
    }

    // Make the link & update pivots
    index_t const id = add_link(grade, prnt, chld, prnt_root, chld_root);
    prnt_it = pivots[prnt_root].emplace_hint(prnt_it, grade.distance, id);
    chld_it = pivots[chld_root].emplace_hint(chld_it, grade.distance, id);

    // Delay inaccessible pivots
    traverse_up(pivots, delay_queue, prnt_it, chld_it, prnt_root, chld_root);
  }
  [[nodiscard]] std::tuple<pivot_it_t, index_t, index_t> find_pivot(
      std::vector<pivot_map_t> &pivots, bigrade_t const &grade, index_t root
  ) {
    index_t id = root;
    pivot_it_t pivot_it = pivots[root].lower_bound(grade.distance);
    if (pivot_it != pivots[root].end()) {
      check_ancestor(pivots, pivot_it, root, grade.distance);
      id = pivot_it->second;
    }
    return std::make_tuple(pivot_it, id, root);
  }
  void traverse_up(
      std::vector<pivot_map_t> &pivots, delay_queue_t &delay_queue,
      pivot_it_t const parent_pivot_it, pivot_it_t const child_pivot_it,
      index_t parent_root, index_t child_root
  ) {
    // Track the previously created pivot
    index_t const delaying_parent = parent_root;
    index_t const delaying_child = child_root;
    index_t prev_root = parent_root;

    // Prepare iterators to larger distances
    pivot_reverse_it_t child_it = std::make_reverse_iterator(child_pivot_it);
    pivot_reverse_it_t parent_it = std::make_reverse_iterator(parent_pivot_it);
    pivot_reverse_it_t child_end = pivots[child_root].rend();
    pivot_reverse_it_t parent_end = pivots[parent_root].rend();

    // Iterate over both sides, processing lowest distance first.
    //   Pivot erasure can invalidate (reverse) iterators. The loop always
    //   ends after an erasure. So no unsafe accesses happen and the iterators
    //   are left potentially invalid.
    while (child_it != child_end or parent_it != parent_end) {
      // One side empty
      if (child_it == child_end) {
        while (parent_it != parent_end) {
          index_t prev_parent_root = parent_root;
          check_ancestor(pivots, parent_it, parent_end, parent_root);
          if (delay_parent_merge(
                  pivots, delay_queue, prev_root, (++parent_it).base(),
                  parent_root, prev_parent_root, child_root
              ))
            break;
        }
        break;
      }
      if (parent_it == parent_end) {
        while (child_it != child_end) {
          index_t prev_child_root = child_root;
          check_ancestor(pivots, child_it, child_end, child_root);
          if (delay_child_merge(
                  pivots, delay_queue, prev_root, (++child_it).base(),
                  child_root, parent_root, prev_child_root
              ))
            break;
        }
        break;
      }
      // Found the same column; erase it and stop.
      if (child_it->first == parent_it->first) {
        check_ancestor(pivots, parent_it, parent_end, parent_root);

        // Erase pivots & delayed merges at this distance
        pivot_it_t base_it = (++parent_it).base();
        delayed_merge_t const merge{base_it, child_root, prev_root};
        queue_it_t it = delay_queue.lower_bound(merge);
        while (it != delay_queue.end() and
               it->parent_it->first == base_it->first)
          delay_queue.erase(it++);
        erase_pivot(pivots, base_it, parent_root);
        break;
      }
      // Process lowest distance first
      if (child_it->first < parent_it->first) {
        index_t prev_child_root = child_root;
        check_ancestor(pivots, child_it, child_end, child_root);
        if (delay_child_merge(
                pivots, delay_queue, prev_root, (++child_it).base(), child_root,
                parent_root, prev_child_root
            ))
          break;
      } else {
        index_t prev_parent_root = parent_root;
        check_ancestor(pivots, parent_it, parent_end, parent_root);
        if (delay_parent_merge(
                pivots, delay_queue, prev_root, (++parent_it).base(),
                parent_root, prev_parent_root, child_root
            ))
          break;
      }
    }
  }
  void check_ancestor(
      std::vector<pivot_map_t> &pivots, pivot_it_t &pivot_it, index_t &root,
      grade_t const distance
  ) {
    while (true) {
      // No ancestor if roots are the same
      index_t new_root = parent_root_of(pivot_it->second);
      if (new_root == root)
        break;

      // Ancestor must exist otherwise
      root = new_root;
      pivot_it = pivots[root].lower_bound(distance);
    }
  }
  void check_ancestor(
      std::vector<pivot_map_t> &pivots, pivot_reverse_it_t &pivot_it,
      pivot_reverse_it_t &end_it, index_t &root
  ) {
    grade_t const distance = pivot_it->first;
    while (true) {
      // No ancestor if roots are the same
      index_t new_root = parent_root_of(pivot_it->second);
      if (new_root == root)
        break;

      // Ancestor must exist otherwise
      root = new_root;
      end_it = pivots[root].rend();
      pivot_it = std::make_reverse_iterator(pivots[root].upper_bound(distance));
    }
  }
  void erase_pivot(
      std::vector<pivot_map_t> &pivots, pivot_it_t const pivot_it, index_t root
  ) {
    auto [distance, link] = *pivot_it;
    pivots[root].erase(pivot_it);

    // Visit descendants
    while (true) {
      // Visit current link
      index_t child_root = child_root_of(link);
      if (child_root != root)
        pivots[child_root].erase(distance);

      // Downstream links cannot have same distance if either side is a point!
      index_t parent = parent_of(link);
      index_t child = child_of(link);
      if (not is_link(parent) or not is_link(child))
        break;

      // Go down child if it has the same distance grade.
      if (distance_grade_of(child) == distance) {
        link = child;
        root = child_root;
        continue;
      }

      // Go down parent if it has the same distance grade.
      if (distance_grade_of(parent) == distance) {
        link = parent;
        continue;
      }

      // No more links.
      break;
    }
  }
  [[nodiscard]] bool delay_parent_merge(
      std::vector<pivot_map_t> &pivots, delay_queue_t &delay_queue,
      index_t &prev_root, pivot_it_t const pivot_it, index_t const pivot_root,
      index_t const parent_root, index_t const child_root
  ) {
    // Construct to-be-delayed merge
    delayed_merge_t merge{pivot_it, pivot_root, prev_root};
    prev_root = std::min(prev_root, pivot_root);

    // Iterate over the equal range
    queue_it_t it = delay_queue.lower_bound(merge);
    while (it != delay_queue.end() and it->parent_it->first == pivot_it->first
    ) {
      // Update child root when delaying parents match
      if (it->child_root == parent_root) {
        it->child_root = merge.child_root;
        return false;
      }

      // Remove delayed merge when delayed parent matches child
      if (it->child_root == child_root) {
        delay_queue.erase(it);
        erase_pivot(pivots, merge.parent_it, pivot_root);
        return true;
      }
      ++it;
    }
    delay_queue.insert(it, merge);
    return false;
  }
  [[nodiscard]] bool delay_child_merge(
      std::vector<pivot_map_t> &pivots, delay_queue_t &delay_queue,
      index_t &prev_root, pivot_it_t const pivot_it, index_t const pivot_root,
      index_t const parent_root, index_t const child_root
  ) {
    // Construct to-be-delayed merge
    delayed_merge_t merge{pivot_it, pivot_root, prev_root};
    prev_root = std::min(prev_root, pivot_root);

    // Iterate over the equal range
    queue_it_t it = delay_queue.lower_bound(merge);
    while (it != delay_queue.end() and it->parent_it->first == pivot_it->first
    ) {
      // Remove (child-delayed) duplicates when parents match
      if (it->child_root == parent_root) {
        delay_queue.erase(it);
        erase_pivot(pivots, merge.parent_it, pivot_root);
        return true;
      }
      // Update merge when its delaying parent matches our child
      if (it->child_root == child_root) {
        it->child_root = merge.child_root;
        // it->delaying_parent = delaying_parent;
        return false;
      }
      ++it;
    }
    delay_queue.insert(it, merge);
    return false;
  }
};

// Deduction guides
template <detail::graded_range link_range_t>
  requires std::ranges::sized_range<link_range_t>
linkage_hierarchy_t(std::from_range_t, link_range_t &&) -> linkage_hierarchy_t<
    typename detail::template_type<std::ranges::range_value_t<
        decltype(std::views::values(std::declval<link_range_t>()))>>::type,
    typename detail::template_type<std::ranges::range_value_t<
        decltype(std::views::keys(std::declval<link_range_t>()))>>::type>;

template <
    std::ranges::sized_range grade_range_t,
    std::ranges::sized_range edge_range_t>
linkage_hierarchy_t(grade_range_t, edge_range_t, std::size_t)
    -> linkage_hierarchy_t<
        typename detail::template_type<
            std::ranges::range_value_t<edge_range_t>>::type,
        typename detail::template_type<
            std::ranges::range_value_t<grade_range_t>>::type>;

}  // namespace bppc

#endif  // BIPERSCAN_LIB_LINKAGE_HIERARCHY_H
