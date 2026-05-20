cmake_minimum_required(VERSION 3.15)
project(CPUDiffusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Пути для библиотек
set(LIB_DIR ${CMAKE_SOURCE_DIR}/../lib/builds)
link_directories(${LIB_DIR})
set(LIB_SRC_DIR ${CMAKE_SOURCE_DIR}/../lib/src)
file(GLOB LIB_SUBDIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true RELATIVE ${LIB_SRC_DIR} ${LIB_SRC_DIR}/*)

# Создаём цели для библиотек
set(CMAKE_JOB_POOLS "lib_build_pool=1")

set(LIB_NAMES)
foreach(SUBDIR ${LIB_SUBDIRS})
    if(IS_DIRECTORY ${LIB_SRC_DIR}/${SUBDIR})
        list(APPEND LIB_NAMES ${SUBDIR})

        add_custom_target(build_lib_${SUBDIR}
                COMMAND $(MAKE) clean -C ${LIB_SRC_DIR}/${SUBDIR}
                COMMAND $(MAKE) all -C ${LIB_SRC_DIR}/${SUBDIR}
                COMMENT "Building library '${SUBDIR}'"
        )

        list(APPEND BUILD_TARGETS build_lib_${SUBDIR})
    endif()
endforeach()

# Итоговая цель для всех либ
add_custom_target(build_libs ALL
        DEPENDS ${BUILD_TARGETS}
        COMMENT "Building all external libraries"
)

# Сборка программы
set(SRC_DIR ${CMAKE_SOURCE_DIR}/../src)
file(GLOB_RECURSE PROGRAM_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#        ${SRC_DIR}/*.cpp
)
add_executable(${PROJECT_NAME} ${PROGRAM_SOURCES})

# Подключение библиотек
include_directories(${CMAKE_SOURCE_DIR}/../lib/includes)
add_dependencies(${PROJECT_NAME} build_libs)
target_link_libraries(${PROJECT_NAME}
        cpudiff
)

# Подготавливаем среду для исполняемого файла
set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "${LIB_DIR}"
        INSTALL_RPATH "\$ORIGIN/../lib/builds"
)
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
