#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/nuclear_mpm.hpp>

TEST_CASE("smoke: public header exposes the nuclear_mpm namespace", "[smoke]") {
    REQUIRE(nuclear_mpm::library_version_major == 1);
}
