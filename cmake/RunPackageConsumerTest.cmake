if(NOT DEFINED NUCLEAR_MPM_BINARY_DIR OR NOT DEFINED NUCLEAR_MPM_SOURCE_DIR)
  message(FATAL_ERROR "Package consumer test requires source and binary directories")
endif()

set(test_root "${NUCLEAR_MPM_BINARY_DIR}/package-consumer-test")
set(install_prefix "${test_root}/install")
set(consumer_build "${test_root}/build")
file(REMOVE_RECURSE "${test_root}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${NUCLEAR_MPM_BINARY_DIR}/_deps/eigen-build"
    --prefix "${install_prefix}"
  RESULT_VARIABLE eigen_install_result
  OUTPUT_VARIABLE eigen_install_output
  ERROR_VARIABLE eigen_install_error
)
if(NOT eigen_install_result EQUAL 0)
  message(FATAL_ERROR "Eigen install failed:\n${eigen_install_output}\n${eigen_install_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${NUCLEAR_MPM_BINARY_DIR}" --prefix "${install_prefix}"
    --component NuclearMPM
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "NuclearMPM install failed:\n${install_output}\n${install_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${NUCLEAR_MPM_SOURCE_DIR}/tests/package_consumer"
    -B "${consumer_build}"
    -GNinja
    "-DCMAKE_PREFIX_PATH=${install_prefix}"
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error
)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "Package consumer configure failed:\n${configure_output}\n${configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}"
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error
)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "Package consumer build failed:\n${build_output}\n${build_error}")
endif()
