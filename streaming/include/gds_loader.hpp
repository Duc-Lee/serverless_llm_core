#pragma once
#include <string>
#include <cufile.h>
#include <cuda.h>
#include <stdexcept>

namespace hydra {

class GDSLoader {
public:
    GDSLoader();
    ~GDSLoader();

    void load_weights_async(const std::string& filepath, 
                            CUdeviceptr mapped_virtual_ptr, 
                            size_t offset, size_t size,
                            CUstream stream = 0);
    
    void synchronize(CUstream stream = 0);
};

} // namespace hydra
