cmake_minimum_required(VERSION 3.15)
project(CPUDiffusion)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- пути ----------
set(LIB_SRC_DIR      ${CMAKE_SOURCE_DIR}/../lib/src)
set(LIB_BUILD_DIR    ${CMAKE_SOURCE_DIR}/../lib/builds)
set(LIB_INCLUDES_DIR ${CMAKE_SOURCE_DIR}/../lib/includes)

# подключаем функцию для сборки библиотек
include(${CMAKE_SOURCE_DIR}/../lib/BuildLibrary.cmake)

# ---------- сборка всех библиотек ----------
file(GLOB LIB_SUBDIRS LIST_DIRECTORIES true ${LIB_SRC_DIR}/*)

set(ALL_LIB_TARGETS)       # цели самих библиотек
set(ALL_COPY_TARGETS)      # цели копирования их заголовков

foreach(LIB_PATH ${LIB_SUBDIRS})
    if(IS_DIRECTORY ${LIB_PATH} AND EXISTS ${LIB_PATH}/CMakeLists.txt)
        get_filename_component(LIB_NAME ${LIB_PATH} NAME)
        add_subdirectory(${LIB_PATH} ${CMAKE_BINARY_DIR}/libs/${LIB_NAME})

        if(LIB_TARGET)
            list(APPEND ALL_LIB_TARGETS ${LIB_TARGET})
        endif()
        if(COPY_TARGET)
            list(APPEND ALL_COPY_TARGETS ${COPY_TARGET})
        endif()

        unset(LIB_TARGET)
        unset(COPY_TARGET)
    endif()
endforeach()

# ---------- кастомная очистка перед сборкой ----------
file(GLOB_RECURSE FILES_TO_CLEAN CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/*repr"
        "${CMAKE_SOURCE_DIR}/*.traw"
        "${CMAKE_SOURCE_DIR}/*.dot"
        "${CMAKE_SOURCE_DIR}/*.svg"
        "${CMAKE_SOURCE_DIR}/out/*"
)

if(FILES_TO_CLEAN)
    add_custom_target(workflow_clean ALL
            COMMAND ${CMAKE_COMMAND} -E rm -f ${FILES_TO_CLEAN}
            COMMENT "Cleaning generated files..."
    )
else()
    add_custom_target(workflow_clean ALL)
endif()

# ---------- основная программа ----------
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${LIB_INCLUDES_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ALL_LIB_TARGETS})

# кастомная очистка
add_dependencies(${PROJECT_NAME} workflow_clean)

# чтобы заголовки всегда были актуальны перед компиляцией
foreach(CT ${ALL_COPY_TARGETS})
    add_dependencies(${PROJECT_NAME} ${CT})
endforeach()

# ---------- настройки запуска ----------
set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH           "${LIB_BUILD_DIR}"
        INSTALL_RPATH         "\$ORIGIN/../lib/builds"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)