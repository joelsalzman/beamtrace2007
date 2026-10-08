// Minimal test harness.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace bttest {

struct TestCase {
  const char* name;
  void (*fn)();
};
std::vector<TestCase>& registry();
extern int checkFailures;

struct Reg {
  Reg(const char* n, void (*f)()) { registry().push_back({n, f}); }
};

}  // namespace bttest

#define TEST(name)                                  \
  static void name();                               \
  static bttest::Reg reg_##name(#name, name);       \
  static void name()

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      ++bttest::checkFailures;                                                      \
      fprintf(stderr, "    CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                               \
  } while (0)

#define CHECK_MSG(cond, ...)                                                        \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      ++bttest::checkFailures;                                                      \
      fprintf(stderr, "    CHECK failed at %s:%d: %s: ", __FILE__, __LINE__, #cond); \
      fprintf(stderr, __VA_ARGS__);                                                 \
      fprintf(stderr, "\n");                                                        \
    }                                                                               \
  } while (0)
