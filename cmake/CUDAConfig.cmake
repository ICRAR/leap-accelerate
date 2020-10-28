
# CUDA Options
if(CUDA_VERSION_STRING VERSION_GREATER 10)
  # 2829 annotation on a defaulted function is ignored
  # 3057 annotation is ignored on a function that is explicitly defaulted
  # 2929 annotation is ignored on a function that is explicitly defaulted
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=3057,2929">)
else()
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=2829">)
endif()