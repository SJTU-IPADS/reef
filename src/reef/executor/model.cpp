
#include <assert.h>
#include <sstream>
#include "reef/executor/model.h"
#include "reef/util/json.h"

namespace reef {
namespace executor {

using reef::util::JsonObject;
using reef::util::JsonParser;

Model* Model::from_json(const char* json_file) {
    std::ifstream fs(json_file);
    std::string tmp, str = "";

    while (getline(fs, tmp)) str += tmp;
    fs.close();

    JsonObject* jobj = JsonParser::parse(str);

    Model* m = new Model;
    for (auto sinfo : jobj->mval["storage"]->lval) {
        m->storage.push_back(StorageInfo{
            sinfo->mval["name"]->sval,
            sinfo->mval["size"]->ival,
            sinfo->mval["stype"]->sval
        });
    }

    for (auto kinfo : jobj->mval["kernels"]->lval) {
        KernelInfo k;

        k.name = kinfo->mval["name"]->sval;
        for (auto arg : kinfo->mval["args"]->lval)
            k.args.push_back(arg->ival);
        
        assert(kinfo->mval["launch_params"]->lval.size() == 6);
        for (int i = 0; i < 6; i++) 
            k.launch_params[i] = kinfo->mval["launch_params"]->lval[i]->ival;
            
        m->kernels.push_back(k);
    }

    for (auto arg : jobj->mval["args"]->lval) {
        m->args.push_back(arg->ival);
    }

    for (auto shared_memory : jobj->mval["shared_memory"]->mval) {
        m->shared_memory[shared_memory.first] = shared_memory.second->ival;
    }
    delete jobj;

    return m;
}

ModelProfile* ModelProfile::from_json(const char* json_file) {
    std::ifstream fs(json_file);
    std::string tmp, str = "";

    while (getline(fs, tmp)) str += tmp;
    fs.close();

    JsonObject* jobj = JsonParser::parse(str);
    ModelProfile* model_profile = new ModelProfile;
    model_profile->model_latency = jobj->mval["model_latency"]->ival;

    for (auto &pair : jobj->mval["kernel_latency"]->mval) {
        const std::string& kernel_name = pair.first;
        KernelProfile profile;
        auto kernel_profile = pair.second->mval;
        profile.total_latency = kernel_profile["total_latency"]->ival;
        for (auto value : kernel_profile["latency"]->lval) {
            profile.latency.push_back(value->ival);
        }
        model_profile->kernel_latency.insert({kernel_name, profile});
    }
    delete jobj;
    return model_profile;
}

#define PARAM_MAGIC "TVM_MODEL_PARAMS"

ModelParam* ModelParamParser::parse_from_file(const char* param_file) {
    FILE* fp;
    fp = fopen(param_file, "rb"); 
    char magic[sizeof(PARAM_MAGIC)];
    size_t res = fread(magic, sizeof(char), sizeof(PARAM_MAGIC), fp);
    assert(res == sizeof(PARAM_MAGIC));
    assert(std::string(magic) == PARAM_MAGIC);
    
    uint64_t params_size;
    res = fread(&params_size, sizeof(uint64_t), 1, fp);
    assert(res == 1);
    assert(params_size != 0);

    ModelParam* params = new ModelParam(params_size);
    for (uint64_t i = 0; i < params_size; i++) {
        char key_buf[256];
        uint64_t key_len = 0;
        while(true) {
            char c;
            res = fread(&c, sizeof(char), 1, fp);
            assert(res == 1);
            key_buf[key_len] = c;
            key_len++;
            if (c == '\0') break;
        }
        std::string key(key_buf);
        uint64_t array_size;
        res = fread(&array_size, sizeof(uint64_t), 1, fp);
        assert(res == 1);
        assert(array_size != 0);
        std::vector<float> array(array_size);
        array.resize(array_size);
        res = fread(array.data(), sizeof(float), array_size, fp);
        assert(res == array_size);
        params->insert({key, array});
    }
    return params;
}

std::string ModelProfile::to_json() {
    std::ostringstream ss;
    
    ss << "{\"model_latency\":" << model_latency << ",\"kernel_latency\":{";
    size_t i = 0;
    for (auto pair : this->kernel_latency) {
        ss << "\"" << pair.first << "\":{\"total_latency\":" << pair.second.total_latency << ", \"latency\":[" ;
        
        size_t j = 0;
        for (auto latency : pair.second.latency) {
            ss << latency;
            j++;
            if (j != pair.second.latency.size()) {
                ss << ",";
            }
        }
        ss << "]}";
        i++;
        if (i != this->kernel_latency.size()) ss << ",";
    }
    ss << "}}";
    return ss.str();
}

size_t Model::get_stype_size(std::string &stype) {
    if (stype == "float32") return 4;
    if (stype == "int64") return 8;
    if (stype == "byte") return 1;
    if (stype == "uint1") return 1;
    if (stype == "int32") return 4;
    std::cout << stype << " is undefined" << std::endl;
    assert(false);
    return 0;
}

} // namespace executor
} // namepsace reef