#ifndef LLM_ENGINE_BPE_TOKENIZER_H
#define LLM_ENGINE_BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <regex>

namespace llm {

/**
 * @brief Hash函数用于std::pair
 */
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

/**
 * @brief BPE (Byte Pair Encoding) Tokenizer for GPT-2
 *
 * GPT-2使用Byte-level BPE分词器，与BERT的WordPiece不同：
 *
 * **关键特性：**
 * 1. Byte-level编码：
 *    - 将所有Unicode字符映射到256个字节
 *    - 保证可以处理任何文本（无UNK）
 *
 * 2. BPE合并：
 *    - 从vocab和merges文件学习
 *    - 贪婪地合并最频繁的字节对
 *
 * 3. 预分词（Pre-tokenization）：
 *    - 按空格、标点分割
 *    - 保留空格信息（作为Ġ前缀）
 *
 * **与WordPiece的区别：**
 * ```
 * WordPiece (BERT):  "playing" -> ["play", "##ing"]
 * BPE (GPT-2):       "playing" -> ["play", "ing"]
 * ```
 *
 * **特殊Token：**
 * - <|endoftext|>: 文本结束标记（ID: 50256）
 *
 * **使用示例：**
 * ```cpp
 * BPETokenizer tokenizer;
 * tokenizer.load("gpt2-vocab.json", "gpt2-merges.txt");
 *
 * // Encode
 * std::vector<int> ids = tokenizer.encode("Hello, world!");
 *
 * // Decode
 * std::string text = tokenizer.decode(ids);
 * ```
 *
 * **文件格式：**
 * - vocab.json: {"token": id, ...}
 * - merges.txt: 每行一个合并规则，如"Ġ h" -> "Ġh"
 */
class BPETokenizer {
public:
    /**
     * @brief 默认构造函数
     */
    BPETokenizer();

    /**
     * @brief 从文件加载tokenizer
     *
     * @param vocab_path vocab.json路径
     * @param merges_path merges.txt路径
     */
    void load(const std::string& vocab_path, const std::string& merges_path);

    /**
     * @brief 编码文本为token IDs
     *
     * @param text 输入文本
     * @return Token IDs
     */
    std::vector<int> encode(const std::string& text) const;

    /**
     * @brief 解码token IDs为文本
     *
     * @param ids Token IDs
     * @return 解码的文本
     */
    std::string decode(const std::vector<int>& ids) const;

    /**
     * @brief 获取词汇表大小
     */
    size_t vocab_size() const { return vocab_.size(); }

    /**
     * @brief 获取特殊token ID
     */
    int eos_token_id() const { return eos_token_id_; }

    /**
     * @brief 将单个token转换为ID
     *
     * @param token Token字符串
     * @return Token ID，如果不存在返回-1
     */
    int token_to_id(const std::string& token) const;

    /**
     * @brief 将ID转换为token
     *
     * @param id Token ID
     * @return Token字符串
     */
    std::string id_to_token(int id) const;

private:
    // Vocab: token -> id
    std::unordered_map<std::string, int> vocab_;

    // Reverse vocab: id -> token
    std::unordered_map<int, std::string> reverse_vocab_;

    // BPE merges: (token1, token2) -> rank
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges_;

    // Byte encoder/decoder (字节<->字符映射)
    std::unordered_map<int, std::string> byte_encoder_;
    std::unordered_map<std::string, int> byte_decoder_;

    // 特殊token IDs
    int eos_token_id_ = -1;  // <|endoftext|>

    // Pre-tokenization pattern (GPT-2正则表达式)
    std::regex pat_;

    /**
     * @brief 初始化字节编码器
     *
     * GPT-2使用特殊的字节到Unicode映射，避免控制字符
     */
    void init_byte_encoder();

    /**
     * @brief 将字符串转换为字节序列
     *
     * @param text 输入文本
     * @return 字节token序列
     */
    std::vector<std::string> bytes_encode(const std::string& text) const;

    /**
     * @brief 将字节token序列转换回字符串
     *
     * @param tokens 字节tokens
     * @return 解码的文本
     */
    std::string bytes_decode(const std::vector<std::string>& tokens) const;

    /**
     * @brief BPE核心算法
     *
     * 对一个单词进行BPE分词
     *
     * @param word 单词（字节序列）
     * @return BPE tokens
     */
    std::vector<std::string> bpe(const std::string& word) const;

    /**
     * @brief 获取字符对
     *
     * @param word 分词后的单词
     * @return 所有相邻字符对
     */
    std::vector<std::pair<std::string, std::string>> get_pairs(
        const std::vector<std::string>& word) const;

    /**
     * @brief 从文件加载vocab
     *
     * @param vocab_path vocab.json路径
     */
    void load_vocab(const std::string& vocab_path);

    /**
     * @brief 从文件加载merges
     *
     * @param merges_path merges.txt路径
     */
    void load_merges(const std::string& merges_path);
};

} // namespace llm

#endif // LLM_ENGINE_BPE_TOKENIZER_H
