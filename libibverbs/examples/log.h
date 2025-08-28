#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

// 日志等级定义
#define LOG_LEVEL_DEBUG   0
#define LOG_LEVEL_INFO    1
#define LOG_LEVEL_WARNING 2
#define LOG_LEVEL_ERROR   3

// 获取当前日志等级
static inline int get_log_level(void) {
    const char *level_str = getenv("VERBS_LOG_LEVEL");
    if (!level_str) {
        return LOG_LEVEL_WARNING; // 默认WARNING级别
    }
    
    if (strcmp(level_str, "0") == 0 || strcasecmp(level_str, "debug") == 0) {
        return LOG_LEVEL_DEBUG;
    } else if (strcmp(level_str, "1") == 0 || strcasecmp(level_str, "info") == 0) {
        return LOG_LEVEL_INFO;
    } else if (strcmp(level_str, "2") == 0 || strcasecmp(level_str, "warning") == 0) {
        return LOG_LEVEL_WARNING;
    } else if (strcmp(level_str, "3") == 0 || strcasecmp(level_str, "error") == 0) {
        return LOG_LEVEL_ERROR;
    }
    
    return LOG_LEVEL_WARNING; // 默认WARNING级别
}

static inline int get_current_log_level(void) {
    static int cached_log_level = -1;
    if (cached_log_level == -1) {
        cached_log_level = get_log_level();
    }

    return cached_log_level;
}

// 获取当前时间字符串
static inline const char* get_time_str(void) {
    static char time_buf[64];
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);
    return time_buf;
}

// 获取文件名（不包含路径）
static inline const char* get_filename(const char* filepath) {
    const char *filename = strrchr(filepath, '/');
    return filename ? filename + 1 : filepath;
}

// 统一的日志打印函数
static inline void log_print(const char* level, const char* file, const char* func, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[%s][%s][%s:%s:%d] ", get_time_str(), level, get_filename(file), func, line);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

// 日志宏定义
#define logd(fmt, ...) do { \
    if (get_current_log_level() <= LOG_LEVEL_DEBUG) { \
        log_print("DEBUG", __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__); \
    } \
} while(0)

#define logi(fmt, ...) do { \
    if (get_current_log_level() <= LOG_LEVEL_INFO) { \
        log_print("INFO ", __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__); \
    } \
} while(0)

#define logw(fmt, ...) do { \
    if (get_current_log_level() <= LOG_LEVEL_WARNING) { \
        log_print("WARN ", __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__); \
    } \
} while(0)

#define loge(fmt, ...) do { \
    if (get_current_log_level() <= LOG_LEVEL_ERROR) { \
        log_print("ERROR", __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__); \
    } \
} while(0)

#endif // LOG_H
