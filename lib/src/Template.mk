ifndef LIB_NAME
    $(error LIB_NAME is not defined in library Makefile!)
endif
ifndef HDRS
    $(error HDRS is not defined in library Makefile!)
endif

CXX = g++
CXXFLAGS += -DLIBRARY_BUILDING -fPIC -Wall -O2 -I. -I..


BUILD_DIR = ../../builds
INCLUDES_DIR = ../../includes/$(LIB_NAME)

LIB_EXT = so
LDFLAGS += -shared
SONAME_FLAG = -Wl,-soname,lib$(LIB_NAME).$(LIB_EXT)

TARGET = $(BUILD_DIR)/lib$(LIB_NAME).$(LIB_EXT)

ifneq ($(strip $(SRCS)),)
    OBJS = $(addsuffix .o, $(SRCS))
    BUILD_LIB = $(TARGET)
else
    OBJS =
    BUILD_LIB =
endif


all: copy_headers $(BUILD_LIB)

ifneq ($(strip $(OBJS)),)
$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $(SONAME_FLAG) -o $@ $^

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

%pp.o: %pp
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif


copy_headers:
ifneq ($(strip $(HDRS)),)
	mkdir -p $(INCLUDES_DIR)
	@for f in $(HDRS); do \
		d="$(INCLUDES_DIR)/$$(dirname "$$f")"; \
		mkdir -p "$$d"; \
		cp "$$f" "$$d/"; \
	done
endif


clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf $(INCLUDES_DIR)

.PHONY: all clean copy_headers