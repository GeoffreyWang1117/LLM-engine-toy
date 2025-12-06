CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I./include
LDFLAGS =

# 源文件
SOURCES = src/matrix.cpp src/layers.cpp \
          src/transformer_block.cpp src/positional_encoding.cpp src/embedding.cpp \
          src/bert_encoder.cpp src/bert_model.cpp src/safetensors.cpp src/weight_loader.cpp \
          src/vocab.cpp src/bert_tokenizer.cpp \
          src/kv_cache.cpp src/causal_attention.cpp src/gpt_decoder_block.cpp src/sampler.cpp \
          src/bpe_tokenizer.cpp src/gpt2_model.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# 目标
TARGET = build/simple_example
TUTORIAL_MATRIX = build/tutorial_matrix
TUTORIAL_LAYERS = build/tutorial_layers
TUTORIAL_ALPHA = build/tutorial_alpha
TUTORIAL_BERT = build/tutorial_bert
TUTORIAL_WEIGHT_LOADING = build/tutorial_weight_loading
TEST_SAFETENSORS = build/test_safetensors
TEST_VOCAB = build/test_vocab
END_TO_END = build/end_to_end_inference
TEST_VARIANTS = build/test_bert_variants
TEST_CAUSAL_ATTN = build/test_causal_attention
TEST_GPT_DECODER = build/test_gpt_decoder
TEST_SAMPLER = build/test_sampler
TEST_BPE = build/test_bpe_tokenizer
TEST_GPT2 = build/test_gpt2_model
GPT2_GENERATION = build/gpt2_text_generation
TEST_REP_PENALTY = build/test_repetition_penalty
TEST_GEN_CONFIG = build/test_generation_config
PERF_PROFILING = build/performance_profiling

# 默认目标
all: $(TARGET) $(TUTORIAL_MATRIX) $(TUTORIAL_ALPHA) $(TUTORIAL_BERT)

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

$(TUTORIAL_BERT): $(OBJECTS) examples/tutorial_bert.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/tutorial_bert.cpp -o $(TUTORIAL_BERT) $(LDFLAGS)

$(TEST_SAFETENSORS): $(OBJECTS) examples/test_safetensors.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_safetensors.cpp -o $(TEST_SAFETENSORS) $(LDFLAGS)

$(TUTORIAL_WEIGHT_LOADING): $(OBJECTS) examples/tutorial_weight_loading.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/tutorial_weight_loading.cpp -o $(TUTORIAL_WEIGHT_LOADING) $(LDFLAGS)

$(TEST_VOCAB): $(OBJECTS) examples/test_vocab.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_vocab.cpp -o $(TEST_VOCAB) $(LDFLAGS)

$(END_TO_END): $(OBJECTS) examples/end_to_end_inference.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/end_to_end_inference.cpp -o $(END_TO_END) $(LDFLAGS)

$(TEST_VARIANTS): $(OBJECTS) examples/test_bert_variants.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_bert_variants.cpp -o $(TEST_VARIANTS) $(LDFLAGS)

$(TEST_CAUSAL_ATTN): $(OBJECTS) examples/test_causal_attention.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_causal_attention.cpp -o $(TEST_CAUSAL_ATTN) $(LDFLAGS)

$(TEST_GPT_DECODER): $(OBJECTS) examples/test_gpt_decoder.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_gpt_decoder.cpp -o $(TEST_GPT_DECODER) $(LDFLAGS)

$(TEST_SAMPLER): $(OBJECTS) examples/test_sampler.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_sampler.cpp -o $(TEST_SAMPLER) $(LDFLAGS)

$(TEST_BPE): $(OBJECTS) examples/test_bpe_tokenizer.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_bpe_tokenizer.cpp -o $(TEST_BPE) $(LDFLAGS)

$(TEST_GPT2): $(OBJECTS) examples/test_gpt2_model.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_gpt2_model.cpp -o $(TEST_GPT2) $(LDFLAGS)

$(GPT2_GENERATION): $(OBJECTS) examples/gpt2_text_generation.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/gpt2_text_generation.cpp -o $(GPT2_GENERATION) $(LDFLAGS)

$(TEST_REP_PENALTY): $(OBJECTS) examples/test_repetition_penalty.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_repetition_penalty.cpp -o $(TEST_REP_PENALTY) $(LDFLAGS)

$(TEST_GEN_CONFIG): $(OBJECTS) examples/test_generation_config.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/test_generation_config.cpp -o $(TEST_GEN_CONFIG) $(LDFLAGS)

$(PERF_PROFILING): $(OBJECTS) examples/performance_profiling.cpp | build
	$(CXX) $(CXXFLAGS) $(OBJECTS) examples/performance_profiling.cpp -o $(PERF_PROFILING) $(LDFLAGS)

# 编译源文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJECTS) $(TARGET) $(TUTORIAL_MATRIX) $(TUTORIAL_LAYERS) $(TUTORIAL_ALPHA) $(TUTORIAL_BERT) $(TEST_SAFETENSORS) $(TUTORIAL_WEIGHT_LOADING) $(TEST_VOCAB) $(END_TO_END) $(TEST_VARIANTS) $(TEST_CAUSAL_ATTN) $(TEST_GPT_DECODER) $(TEST_SAMPLER) $(TEST_BPE) $(TEST_GPT2) $(GPT2_GENERATION) $(TEST_REP_PENALTY) $(TEST_GEN_CONFIG) $(PERF_PROFILING)
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

run-bert: $(TUTORIAL_BERT)
	./$(TUTORIAL_BERT)

run-test-safetensors: $(TEST_SAFETENSORS)
	./$(TEST_SAFETENSORS)

run-weight-loading: $(TUTORIAL_WEIGHT_LOADING)
	./$(TUTORIAL_WEIGHT_LOADING)

run-test-vocab: $(TEST_VOCAB)
	./$(TEST_VOCAB)

run-end-to-end: $(END_TO_END)
	./$(END_TO_END)

run-test-variants: $(TEST_VARIANTS)
	./$(TEST_VARIANTS)

run-test-causal-attn: $(TEST_CAUSAL_ATTN)
	./$(TEST_CAUSAL_ATTN)

run-test-gpt-decoder: $(TEST_GPT_DECODER)
	./$(TEST_GPT_DECODER)

run-test-sampler: $(TEST_SAMPLER)
	./$(TEST_SAMPLER)

run-test-bpe: $(TEST_BPE)
	./$(TEST_BPE)

run-test-gpt2: $(TEST_GPT2)
	./$(TEST_GPT2)

run-gpt2-generation: $(GPT2_GENERATION)
	./$(GPT2_GENERATION)

run-test-repetition-penalty: $(TEST_REP_PENALTY)
	./$(TEST_REP_PENALTY)

run-test-generation-config: $(TEST_GEN_CONFIG)
	./$(TEST_GEN_CONFIG)

run-performance-profiling: $(PERF_PROFILING)
	./$(PERF_PROFILING)

.PHONY: all clean run run-matrix run-layers run-alpha run-bert run-test-safetensors run-weight-loading run-test-vocab run-end-to-end run-test-variants run-test-causal-attn run-test-gpt-decoder run-test-sampler run-test-bpe run-test-gpt2 run-gpt2-generation run-test-repetition-penalty run-test-generation-config run-performance-profiling build
