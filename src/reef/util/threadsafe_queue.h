#pragma once

#include <atomic>
#include <mutex>
#include <assert.h>


namespace reef {
// TODO: replace it with a lock-free queue
template<typename Element>
class ThreadSafeQueue {
public:

    enum { Capacity = 10000 };

    ThreadSafeQueue() : _tail(0), _head(0){
        _array.resize(Capacity);
    }   

    virtual ~ThreadSafeQueue() {
    }

    ThreadSafeQueue(const ThreadSafeQueue &queue) = delete;

    ThreadSafeQueue(ThreadSafeQueue && queue) noexcept {
        _tail.store(queue._tail.load());
        _head.store(queue._head.load());
        _array = std::move(queue._array);
    } 

    /* Producer only: updates tail index after setting the element in place */
    bool push(const Element& item)
    {	
        // quick fix: lock the producers
        std::unique_lock<std::mutex> lock(mtx);
        auto current_tail = _tail.load();            
        auto next_tail = increment(current_tail);    
        if(next_tail != _head.load())                         
        {
            _array[current_tail] = item;               
            _tail.store(next_tail);                    
            return true;
        }
        
        return false;  // full queue
    }

    /* Consumer only: updates head index after retrieving the element */
    void pop()
    {
        std::unique_lock<std::mutex> lock(mtx);
        const auto current_head = _head.load();  
        assert(current_head != _tail.load());   // empty queue
        _head.store(increment(current_head)); 
    }

    Element& front()
    {
        std::unique_lock<std::mutex> lock(mtx);
        const auto current_head = _head.load();  
        assert(current_head != _tail.load());   // empty queue
        auto &item = _array[current_head]; 
        return item;
    }

    bool empty() const {
        // std::unique_lock<std::mutex> lock(mtx);
        return (_head.load() == _tail.load());
    }

    bool full() const
    {
        const auto next_tail = increment(_tail.load());
        return (next_tail == _head.load());
    }

private:
    size_t increment(size_t idx) const
    {
        return (idx + 1) % Capacity;
    }
    std::atomic<size_t>  _tail;  
    std::vector<Element> _array;
    std::mutex mtx;
    std::atomic<size_t>  _head; 
};
} // namespace reef