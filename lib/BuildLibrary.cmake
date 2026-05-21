include_guard(GLOBAL)

function(add_dynamic_library)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS COMPILE_FLAGS LINK_LIBRARIES)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_NAME)
        message(FATAL_ERROR "add_dynamic_library: NAME argument is required")
    endif()

    set(LIB_NAME  ${ARG_NAME})
    set(LIB_TARGET ${LIB_NAME})
    set(COPY_TARGET copy_headers_${LIB_NAME})

    # === 1. Цель для сборки динамической библиотеки ===
    if(ARG_SOURCES)
        add_library(${LIB_TARGET} SHARED ${ARG_SOURCES})

        get_filename_component(LIB_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

        # Добавляем директории для подключения заголовков
        target_include_directories(${LIB_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_include_directories(${LIB_TARGET} PRIVATE ${LIB_PARENT_DIR})

        if(ARG_COMPILE_FLAGS)
            target_compile_options(${LIB_TARGET} PRIVATE ${ARG_COMPILE_FLAGS})
        endif()

        if(ARG_LINK_LIBRARIES)
            target_link_libraries(${LIB_TARGET} PRIVATE ${ARG_LINK_LIBRARIES})
        endif()

        set_target_properties(${LIB_TARGET} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${LIB_BUILD_DIR}"
            PREFIX "lib"
            SUFFIX ".so"
        )
    else()
        # Если исходников нет, создаём пустой таргет
        set(LIB_TARGET "")
        # Можно использовать для теста
        # message(WARNING "Library '${LIB_NAME}' has no SOURCES, only headers will be installed")
    endif()

    # === 2. Копирование заголовков ===
    if(ARG_HEADERS)
        # Собираем пути заголовков и команды для копирования
        set(HEADER_SRCS_ABS)
        set(COPY_COMMANDS)

        foreach(HEADER ${ARG_HEADERS})
            if(IS_ABSOLUTE ${HEADER})
                set(HDR_SRC ${HEADER})
            else()
                set(HDR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER})
            endif()

            list(APPEND HEADER_SRCS_ABS ${HDR_SRC})

            # Сохраняем структуру подкаталогов
            file(RELATIVE_PATH HDR_REL ${CMAKE_CURRENT_SOURCE_DIR} ${HDR_SRC})
            set(HDR_DST ${LIB_INCLUDES_DIR}/${LIB_NAME}/${HDR_REL})
            get_filename_component(HDR_DST_DIR ${HDR_DST} DIRECTORY)

            list(APPEND COPY_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E make_directory "${HDR_DST_DIR}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${HDR_SRC}" "${HDR_DST}"
            )
        endforeach()

        add_custom_target(${COPY_TARGET}
            ${COPY_COMMANDS}
            DEPENDS ${HEADER_SRCS_ABS}
            COMMENT "Copying public headers for ${LIB_NAME}"
        )

        if(LIB_TARGET)
            # Библиотека должна ждать окончания копирования
            add_dependencies(${LIB_TARGET} ${COPY_TARGET})
        endif()
    else()
        # Если заголовков нет, создаём пустой таргет
        add_custom_target(${COPY_TARGET})
    endif()

    # === 3. Экспорт имён целей для родительского CMakeLists.txt ===
    set(LIB_TARGET   ${LIB_TARGET}   PARENT_SCOPE)
    set(COPY_TARGET  ${COPY_TARGET}  PARENT_SCOPE)
endfunction()
