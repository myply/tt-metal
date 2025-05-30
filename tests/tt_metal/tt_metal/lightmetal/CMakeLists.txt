set(UNIT_TESTS_LIGHTMETAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_lightmetal.cpp)

add_executable(unit_tests_lightmetal ${UNIT_TESTS_LIGHTMETAL_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_lightmetal)

target_link_libraries(
    unit_tests_lightmetal
    PUBLIC
        test_metal_common_libs
        FlatBuffers::FlatBuffers
)

target_include_directories(
    unit_tests_lightmetal
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set_target_properties(
    unit_tests_lightmetal
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
