#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace reef {
namespace executor {

class StorageInfo {
public:
    std::string name;
    size_t size;
    std::string stype;
};

class KernelInfo {
public:
    std::string name;
    uint32_t launch_params[6];
    std::vector<size_t> args;
};

class Model {
public:
    std::vector<StorageInfo> storage;
    std::vector<KernelInfo> kernels;
    std::vector<uint32_t> args;
    std::unordered_map<std::string, size_t> shared_memory;

public:
    static Model* from_json(const char* json_file);
    static size_t get_stype_size(std::string &stype);
};

class KernelProfile {
public:
    std::vector<int> latency; // microsecond
    int total_latency;
    int estimated_latency(int occupancy, int task_num_per_block);
};

class ModelProfile {
public:
    int model_latency;
    std::unordered_map<std::string, KernelProfile> kernel_latency;
    std::string to_json();
    static ModelProfile* from_json(const char* json_file);
};

typedef std::unordered_map<std::string, std::vector<float>> ModelParam;

class ModelParamParser {
public:
    static ModelParam* parse_from_file(const char* param_file);
};



} // namespace executor
} // namespace reef