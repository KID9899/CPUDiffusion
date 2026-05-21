# Этот файл не относится к библиотеке nlohmann/json
# он написан для интеграции этой библиотеки в текущий проект

file(GLOB_RECURSE NLOHMANN_HPPS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

add_dynamic_library(
    NAME nlohmann
    HEADERS ${NLOHMANN_HPPS}
)

# Пробрасываем имена целей на верхний уровень (обязательно)
set(LIB_TARGET  ${LIB_TARGET}  PARENT_SCOPE)
set(COPY_TARGET ${COPY_TARGET} PARENT_SCOPE)