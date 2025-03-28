#ifndef BIPERSCAN_LIB_MINIMAL_PRESENTATION_ITERATION_STATE_H
#define BIPERSCAN_LIB_MINIMAL_PRESENTATION_ITERATION_STATE_H

#include "minimal_presentation_graph.h"

namespace bppc::detail {

// Iterator that maintains state required for detecting merges in minimal
// presentations.
template <
    templated_view_of<edge_t> edge_view_t,
    templated_view_of<bigrade_t> grade_view_t>
class minpres_iterator_t {
  // Type shorthand
  using index_t =
      typename template_type<std::ranges::range_value_t<edge_view_t>>::type;
  using grade_t =
      typename template_type<std::ranges::range_value_t<grade_view_t>>::type;
  using edge_t = edge_t<index_t>;
  using bigrade_t = bigrade_t<grade_t>;

  // Input state
  minpres_graph_t<edge_view_t, grade_view_t> const &d_graph;
  index_t const d_column_limit{};
  grade_t const d_distance_limit{};

  // Traversal queue
  std::vector<bool> d_enqueued{};
  std::vector<bool> d_visited{};
  std::map<grade_t, index_t> d_queue{};

  // Traversal state
  index_t d_root = 0;
  std::vector<index_t> d_children{};

 public:
  minpres_iterator_t(
      minpres_graph_t<edge_view_t, grade_view_t> const &graph,
      index_t const root, std::optional<index_t> const connecting_point,
      index_t const column_limit, grade_t const distance_limit
  )
      : d_graph(graph),
        d_column_limit(column_limit),
        d_distance_limit(distance_limit),
        d_enqueued(graph.num_points()),
        d_visited(graph.num_points()),
        d_root(root),
        d_children({root}) {
    d_enqueued[root] = true;
    d_visited[root] = true;
    if (connecting_point)
      d_enqueued[*connecting_point] = true;
    enqueue_children(root);
  }

  [[nodiscard]] bool empty() const {
    return d_queue.empty();
  }
  [[nodiscard]] grade_t next_distance() const {
    return d_queue.begin()->first;
  }
  [[nodiscard]] bool collect_children(
      minpres_iterator_t const &other, grade_t const distance_bound
  ) {
    while (!empty() and next_distance() <= distance_bound) {
      index_t const node = next();
      if (other.d_visited[node])
        return true;
      d_visited[node] = true;
      d_children.push_back(node);
      d_root = std::min(node, d_root);
      enqueue_children(node);
    }
    return false;
  }
  [[nodiscard]] index_t root() const {
    return d_root;
  }
  [[nodiscard]] std::size_t num_children() const {
    return d_children.size();
  }
  [[nodiscard]] std::vector<index_t> &&take_children() {
    return std::move(d_children);
  }

 private:
  [[nodiscard]] index_t next() {
    auto it = d_queue.begin();
    index_t const node = it->second;
    d_queue.erase(it);
    return node;
  }
  void enqueue_children(index_t const node) {
    std::vector<index_t> const &child_columns = d_graph.children(node);
    auto upper_limit = std::ranges::distance(
        std::ranges::lower_bound(child_columns, d_column_limit),
        child_columns.end()
    );

    for (index_t const edge_idx :
         std::views::reverse(child_columns) | std::views::drop(upper_limit)) {
      auto const [distance, child] = d_graph.child_at(node, edge_idx);
      if (distance >= d_distance_limit or d_enqueued[child])
        continue;
      d_enqueued[child] = true;
      d_queue.emplace(distance, child);
    }
  }
};

}  // namespace bppc::detail

#endif  // BIPERSCAN_LIB_MINIMAL_PRESENTATION_ITERATION_STATE_H
