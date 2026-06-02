add_dynamic_library(
    NAME cpudiff
    SOURCES
        typeutils/to_float.cpp

        core/graph.cpp
        core/tensor.cpp

        safetensors/reader.cpp

        layers/module.cpp
        layers/linear.cpp
        layers/sequential.cpp
        layers/functions.cpp
        layers/transformer_parts.cpp

        images/png.cpp

        executors/simple.cpp

        models/tokenizer.cpp

    HEADERS
        typeutils/dtypes.h
        typeutils/to_float.h

        core/graph.h
        core/tensor.h
        core/any_tensor.h
        core/operation.h

        safetensors/reader.h

        layers/module.h
        layers/module_list.h
        layers/linear.h
        layers/sequential.h
        layers/functions.h
        layers/transformer_parts.h

        images/png.h

        executors/simple.h

        models/tokenizer.h

        typeutils.h
        core.h
        safetensors.h
        layers.h
        images.h
        executors.h
        models.h

        cpudiff.h

    COMPILE_FLAGS
        -fopenmp
    LINK_LIBRARIES
        -fopenmp
)

# Пробрасываем имена целей на верхний уровень (обязательно)
set(LIB_TARGET  ${LIB_TARGET}  PARENT_SCOPE)
set(COPY_TARGET ${COPY_TARGET} PARENT_SCOPE)