#ifndef LLM_ENGINE_PROFILER_H
#define LLM_ENGINE_PROFILER_H

#include <string>
#include <chrono>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace llm {

/**
 * @brief 简单的性能分析工具
 *
 * 用于测量代码块的执行时间和调用次数
 */
class Profiler {
public:
    struct Stats {
        int count = 0;                  // 调用次数
        double total_ms = 0.0;          // 总时间（毫秒）
        double min_ms = 1e9;            // 最小时间
        double max_ms = 0.0;            // 最大时间

        double avg_ms() const {
            return count > 0 ? total_ms / count : 0.0;
        }
    };

    /**
     * @brief 获取全局Profiler实例
     */
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }

    /**
     * @brief 开始计时
     */
    void start(const std::string& name) {
        start_times_[name] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief 结束计时并记录
     */
    void stop(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto it = start_times_.find(name);

        if (it != start_times_.end()) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end - it->second
            ).count();

            double ms = duration / 1000.0;

            Stats& stats = stats_[name];
            stats.count++;
            stats.total_ms += ms;
            stats.min_ms = std::min(stats.min_ms, ms);
            stats.max_ms = std::max(stats.max_ms, ms);

            start_times_.erase(it);
        }
    }

    /**
     * @brief 获取统计信息
     */
    const std::map<std::string, Stats>& get_stats() const {
        return stats_;
    }

    /**
     * @brief 重置所有统计
     */
    void reset() {
        stats_.clear();
        start_times_.clear();
    }

    /**
     * @brief 打印统计报告
     */
    void print_report(bool sort_by_total = true) const {
        if (stats_.empty()) {
            std::cout << "No profiling data available.\n";
            return;
        }

        // 转换为vector以便排序
        std::vector<std::pair<std::string, Stats>> sorted_stats(
            stats_.begin(), stats_.end()
        );

        if (sort_by_total) {
            std::sort(sorted_stats.begin(), sorted_stats.end(),
                [](const auto& a, const auto& b) {
                    return a.second.total_ms > b.second.total_ms;
                });
        }

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                          Performance Report                              ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        // 表头
        std::cout << std::left << std::setw(35) << "Name"
                  << std::right << std::setw(10) << "Count"
                  << std::setw(12) << "Total(ms)"
                  << std::setw(12) << "Avg(ms)"
                  << std::setw(12) << "Min(ms)"
                  << std::setw(12) << "Max(ms)"
                  << "\n";
        std::cout << std::string(93, '-') << "\n";

        // 数据行
        for (const auto& [name, stats] : sorted_stats) {
            std::cout << std::left << std::setw(35) << name
                      << std::right << std::setw(10) << stats.count
                      << std::setw(12) << std::fixed << std::setprecision(3) << stats.total_ms
                      << std::setw(12) << std::fixed << std::setprecision(3) << stats.avg_ms()
                      << std::setw(12) << std::fixed << std::setprecision(3) << stats.min_ms
                      << std::setw(12) << std::fixed << std::setprecision(3) << stats.max_ms
                      << "\n";
        }

        std::cout << std::string(93, '-') << "\n";

        // 计算总时间
        double total_time = 0.0;
        for (const auto& [_, stats] : stats_) {
            total_time += stats.total_ms;
        }

        std::cout << "Total measured time: " << std::fixed << std::setprecision(3)
                  << total_time << " ms\n\n";
    }

private:
    Profiler() = default;

    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::map<std::string, Stats> stats_;
};

/**
 * @brief RAII风格的计时器
 *
 * 在构造时开始计时，析构时自动结束计时
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name), profiler_(Profiler::instance()) {
        profiler_.start(name_);
    }

    ~ScopedTimer() {
        profiler_.stop(name_);
    }

    // 禁止拷贝
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string name_;
    Profiler& profiler_;
};

// 便利宏
#define PROFILE_SCOPE(name) llm::ScopedTimer _timer_##__LINE__(name)
#define PROFILE_FUNCTION() llm::ScopedTimer _timer_##__LINE__(__FUNCTION__)

} // namespace llm

#endif // LLM_ENGINE_PROFILER_H
