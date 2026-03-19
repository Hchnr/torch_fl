# fl/csrc CMakeLists.txt
file(GLOB_RECURSE FL_CSRC_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aten/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stub.cpp
)
