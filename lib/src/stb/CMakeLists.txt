# Этот файл не относится к библиотеке stb
# он написан для интеграции этой библиотеки в текущий проект

file(GLOB_RECURSE STB_HS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_dynamic_library(
    NAME stb
    HEADERS ${STB_HS}
)

# Пробрасываем имена целей на верхний уровень (обязательно)
set(LIB_TARGET  ${LIB_TARGET}  PARENT_SCOPE)
set(COPY_TARGET ${COPY_TARGET} PARENT_SCOPE)