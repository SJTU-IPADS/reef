#pragma once 

#include <string>

namespace reef {
namespace util {

class SharedMemory {
public:
    SharedMemory(const std::string& __key, size_t __size, bool create=false);
    ~SharedMemory();
    
    void* data();
    size_t size();

private:
    int _fd;
    std::string _key;
    size_t _size;
    void* _data;
    bool _create;

}; 


} // namespace util
} // namespace reef 