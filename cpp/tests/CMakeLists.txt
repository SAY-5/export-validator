include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(export_validator_tests
  test_differ.cpp
  test_json_writer.cpp
  test_npz_loader.cpp
)
target_link_libraries(export_validator_tests PRIVATE
  export_validator_core GTest::gtest GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(export_validator_tests)
