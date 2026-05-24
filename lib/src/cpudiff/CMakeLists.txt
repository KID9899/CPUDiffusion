add_dynamic_library(
    NAME cpudiff
    SOURCES
        typeutils/float.cpp

        core/graph.cpp
        core/tensor.cpp

        safetensors/reader.cpp

        layers/module.cpp
        layers/linear.cpp
        layers/sequential.cpp
        layers/functions.cpp

        images/png.cpp

        executors/simple.cpp

    HEADERS
        typeutils/dtypes.h
        typeutils/float.h

        core/graph.h
        core/tensor.h
        core/operation.h

        safetensors/reader.h

        layers/module.h
        layers/linear.h
        layers/sequential.h
        layers/functions.h

        images/png.h

        executors/simple.h

        typeutils.h
        core.h
        safetensors.h
        layers.h
        images.h
        executors.h

        cpudiff.h

    COMPILE_FLAGS
        -fopenmp
    LINK_LIBRARIES
        -fopenmp
)

# Пробрасываем имена целей на верхний уровень (обязательно)
set(LIB_TARGET  ${LIB_TARGET}  PARENT_SCOPE)
set(COPY_TARGET ${COPY_TARGET} PARENT_SCOPE)