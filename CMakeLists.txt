project(reef)
cmake_minimum_required(VERSION 3.2)

add_compile_options(-std=c++11)
add_definitions(-D__REEF_HIP_GPU__)
add_definitions(-DRESOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}/resource")


# GRPC and Protocol Buffers libraries location
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

# Cmake find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# For grpc 

include("${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake")

get_filename_component(reef_proto "${CMAKE_CURRENT_LIST_DIR}/src/reef/protos/reef.proto" ABSOLUTE)
get_filename_component(reef_proto_path "${reef_proto}" PATH)
set(GRPC_GENERATE_DIR "${PROJECT_SOURCE_DIR}/src/reef/rpc" )
set(reef_proto_srcs "${GRPC_GENERATE_DIR}/reef.pb.cc")
set(reef_proto_hdrs "${GRPC_GENERATE_DIR}/reef.pb.h")
set(reef_grpc_srcs "${GRPC_GENERATE_DIR}/reef.grpc.pb.cc")
set(reef_grpc_hdrs "${GRPC_GENERATE_DIR}/reef.grpc.pb.h")
add_custom_command(
      OUTPUT "${reef_proto_srcs}" "${reef_proto_hdrs}" "${reef_grpc_srcs}" "${reef_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${GRPC_GENERATE_DIR}"
        --cpp_out "${GRPC_GENERATE_DIR}"
        -I "${reef_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${reef_proto}"
      DEPENDS "${reef_proto}")

include_directories("${GRPC_GENERATE_DIR}")

# reef_grpc_proto
add_library(reef_grpc_proto
  ${reef_grpc_srcs}
  ${reef_grpc_hdrs}
  ${reef_proto_srcs}
  ${reef_proto_hdrs})
target_link_libraries(reef_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})


# REEF codes

find_package(hip REQUIRED)

find_package(GTest REQUIRED)

find_package(glog REQUIRED)

SET(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")


set(CMAKE_CXX_FLAGS "-g -O0 ${CMAKE_CXX_FLAGS}")

include_directories(${HIP_INCLUDE_DIRS})

include_directories("${PROJECT_SOURCE_DIR}/src")


AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/reef/client" client)

AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/reef/executor" executor)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/reef/executor/hip" hip_impl)

AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/reef/server" server)

AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/reef/util" util)

AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/reef/test" test)

add_library(reef_util
  ${util}
)

add_library(reef_server_lib
  ${server}
  ${executor}
  ${hip_impl}
)

add_library(reef_client_lib
  ${client}
)

add_executable(unit_test 
  ${test}
)

target_link_libraries(unit_test
    reef_util
    reef_server_lib
    reef_client_lib
    reef_grpc_proto
    glog::glog
    ${GTEST_BOTH_LIBRARIES}
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
    pthread
)

function (add_executable_app app_name app_path)
    add_executable(${app_name}
        ${app_path}
    )
    target_link_libraries(${app_name}
      reef_util
      reef_server_lib
      reef_client_lib
      reef_grpc_proto
      glog::glog
      ${GTEST_BOTH_LIBRARIES}
      ${_REFLECTION}
      ${_GRPC_GRPCPP}
      ${_PROTOBUF_LIBPROTOBUF}
      pthread
    )
endfunction()

add_executable_app(reef_client "${PROJECT_SOURCE_DIR}/src/example/rpc_client.cpp")
add_executable_app(reef_client_cont "${PROJECT_SOURCE_DIR}/src/example/rpc_client_cont.cpp")
add_executable_app(reef_server "${PROJECT_SOURCE_DIR}/src/example/rpc_server.cpp")
