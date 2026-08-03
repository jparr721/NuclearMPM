function(nuclear_mpm_enable_sanitizers target)
  if(NOT NUCLEAR_MPM_ENABLE_SANITIZERS)
    return()
  endif()

  if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # AppleClang's ASan runtime can hang while locating dynamic shadow memory on
    # current arm64 macOS releases. Keep bounded sanitizer validation usable by
    # enabling UBSan here; Clang/GCC platforms retain combined ASan + UBSan.
    set(sanitizer_flags -fsanitize=undefined -fno-omit-frame-pointer)
    target_compile_options(${target} PRIVATE ${sanitizer_flags})
    target_link_options(${target} PRIVATE ${sanitizer_flags})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(sanitizer_flags -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_compile_options(${target} PRIVATE ${sanitizer_flags})
    target_link_options(${target} PRIVATE ${sanitizer_flags})
  else()
    message(WARNING "NuclearMPM sanitizers are not configured for ${CMAKE_CXX_COMPILER_ID}")
  endif()
endfunction()
