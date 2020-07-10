set(TARGET_NAME LeapAccelerateServer)
set(TARGET_EXPORT leapaccelerate-targest)
set(TARGET_TEST LeapAccelerateServerExec)

set(sources
  #Calibrate.cc
  leap_cal_server.cc
  #remote_cal.cc
)

set(private_headers
  #Calibrate.h
  leap_cal_server.h
  #remote_cal.h
)

find_package(Threads)

# add the executable
add_executable(
  ${TARGET_NAME}
  ${sources}
  ${private_headers}
  ${public_headers}
)
target_link_libraries (${TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT} LeapAccelerate)

install (
  TARGETS
    ${TARGET_NAME}
  EXPORT
    ${TARGET_EXPORT}
  DESTINATION
    bin
)

#add_test(NAME ${TARGET_TEST} COMMAND ${TARGET_NAME})
