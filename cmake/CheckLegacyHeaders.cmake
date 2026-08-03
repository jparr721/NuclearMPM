if(NOT DEFINED NUCLEAR_MPM_SOURCE_DIR)
  message(FATAL_ERROR "Legacy header check requires NUCLEAR_MPM_SOURCE_DIR")
endif()

foreach(legacy_header IN ITEMS src/nclr.h src/nclr_math.h)
  if(EXISTS "${NUCLEAR_MPM_SOURCE_DIR}/${legacy_header}")
    message(FATAL_ERROR "Legacy implementation still exists: ${legacy_header}")
  endif()
endforeach()

file(GLOB_RECURSE production_files
  "${NUCLEAR_MPM_SOURCE_DIR}/include/*.h"
  "${NUCLEAR_MPM_SOURCE_DIR}/include/*.hpp"
  "${NUCLEAR_MPM_SOURCE_DIR}/src/*.h"
  "${NUCLEAR_MPM_SOURCE_DIR}/src/*.hpp"
  "${NUCLEAR_MPM_SOURCE_DIR}/src/*.cpp"
)
foreach(production_file IN LISTS production_files)
  file(READ "${production_file}" contents)
  if(contents MATCHES "nclr\\.h|nclr_math\\.h")
    message(FATAL_ERROR "Legacy include remains in ${production_file}")
  endif()
endforeach()
