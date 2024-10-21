#include <cstddef>

namespace bppc::detail {

std::size_t WIDTH = 1;
std::size_t HWIDTH = 1;

void set_width(std::size_t max_value) {
  std::size_t count = 1;
  while (max_value /= 10) {
    ++count;
  }
  WIDTH = count;
}

void set_hwidth(std::size_t max_value) {
  std::size_t count = 1;
  while (max_value /= 10) {
    ++count;
  }
  HWIDTH = count;
}

}  // namespace bppc::detail