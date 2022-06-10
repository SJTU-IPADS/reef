#pragma once

#include <stdlib.h>  
#include <glog/logging.h>
#include <iostream>
#include <memory>

#define DEFAULT_REEF_ADDR "localhost:34543"

#ifndef RESOURCE_DIR
#define RESOURCE_DIR "../resource"
#endif

#define ASSERT(condition)\
     do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << ":" << __LINE__ << std::endl; \
            std::terminate(); \
        } \
    } while (false)

#define ASSERT_STATUS(cmd) ASSERT(cmd == Status::Succ)

#define RETURN_STATUS(cmd) \
{\
    Status s = cmd;\
    if (s != Status::Succ) {\
        LOG(ERROR) << #cmd " error, " << __FILE__ << ":" << __LINE__; \
        return s;\
    }\
}

#define ASSERT_MSG(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << ":" << __LINE__ << " msg: " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)


namespace reef {

enum Status {
    Succ,
    Fail,
    NotFound,
    OutOfRange,
    Full
};

template <typename T>
T align_up(T value, T alignment) {
    T temp = value % alignment;
    return temp == 0? value : value - temp + alignment;
}

template <typename T>
T align_down(T value, T alignment) {
    return value - value % alignment;
}
}

