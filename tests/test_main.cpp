#include <cstring>

#include "core/real.h"
#include "tests/test.h"
#include "util/misc.h"

namespace bttest {
std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}
int checkFailures = 0;
}  // namespace bttest

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int failedTests = 0, ran = 0;
  printf("precision: %s\n", sizeof(bt::Real) == 8 ? "double" : "float");
  for (auto& t : bttest::registry()) {
    if (filter && !strstr(t.name, filter)) continue;
    int before = bttest::checkFailures;
    bt::Timer timer;
    t.fn();
    ++ran;
    bool ok = bttest::checkFailures == before;
    if (!ok) ++failedTests;
    printf("[%s] %s (%.2fs)\n", ok ? " OK " : "FAIL", t.name, timer.seconds());
    fflush(stdout);
  }
  printf("%d/%d tests passed\n", ran - failedTests, ran);
  return failedTests == 0 ? 0 : 1;
}
