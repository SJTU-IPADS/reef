#include "reef/util/shared_memory.h"
#include "reef/util/common.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

namespace reef {
namespace util {

SharedMemory::SharedMemory(
    const std::string& __key, size_t __size, bool create
) : _key(__key), _size(__size), _create(create)
{
    _fd = shm_open(
        __key.c_str(),
        create ? O_CREAT|O_RDWR : O_RDWR,
        0777
    );
    ASSERT(_fd >= 0);
    ASSERT(ftruncate(_fd, _size) >= 0);
    _data = mmap(NULL, _size, PROT_READ|PROT_WRITE, MAP_SHARED, _fd, 0);
    ASSERT(_data != nullptr);
}

SharedMemory::~SharedMemory() {
    close(_fd);
    if (_create) {
        shm_unlink(_key.c_str());
    }
}

void* SharedMemory::data() {
    return _data;
}

size_t SharedMemory::size() {
    return _size;
}

} // namespace util
} // namespace reef 