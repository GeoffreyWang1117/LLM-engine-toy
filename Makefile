CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I./include
LDFLAGS =

# 源文件
SOURCES = src/matrix.cpp src/layers.cpp \
          src/transformer_block.cpp src/positional_encoding.cpp src/embedding.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# 目标
TARGET = build/simple_example
TUTORIAL_MATRIX = build/tutorial_matrix
TUTORIAL_LAYERS = build/tutorial_layers
TUTORIAL_ALPHA = build/tutorial_alpha

# 默认目标
all: $(TARGET) $(TUTORIAL_MATRIX) $(TUTORIAL_ALPHA)

# 创建build目录
build:
	mkdir -p build

# 编译目标
$(TARGET): $(OBJECTS) examples/simple_example.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/simple_example.cpp -o $(TARGET) $(LDFLAGS)

$(TUTORIAL_MATRIX): $(OBJECTS) examples/tutorial_matrix.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/tutorial_matrix.cpp -o $(TUTORIAL_MATRIX) $(LDFLAGS)

$(TUTORIAL_LAYERS): $(OBJECTS) examples/tutorial_layers.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/tutorial_layers.cpp -o $(TUTORIAL_LAYERS) $(LDFLAGS)

$(TUTORIAL_ALPHA): $(OBJECTS) examples/tutorial_alpha.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/tutorial_alpha.cpp -o $(TUTORIAL_ALPHA) $(LDFLAGS)

# 编译源文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJECTS) $(TARGET) $(TUTORIAL_MATRIX) $(TUTORIAL_LAYERS) $(TUTORIAL_ALPHA)
	rm -rf build

# 运行
run: $(TARGET)
	./$(TARGET)

run-matrix: $(TUTORIAL_MATRIX)
	./$(TUTORIAL_MATRIX)

run-layers: $(TUTORIAL_LAYERS)
	./$(TUTORIAL_LAYERS)

run-alpha: $(TUTORIAL_ALPHA)
	./$(TUTORIAL_ALPHA)

.PHONY: all clean run run-matrix run-layers run-alpha build
