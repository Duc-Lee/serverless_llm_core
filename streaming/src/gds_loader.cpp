#include "gds_loader.hpp"
#include <fcntl.h>
#ifdef __linux__
#include <unistd.h>
#endif

#define CUFILE_CHECK(call) \
    do { \
        CUfileError_t status = call; \
        if (status.err != CU_FILE_SUCCESS) { \
            throw std::runtime_error("cuFile API error"); \
        } \
    } while (0)

namespace hydra {

GDSLoader::GDSLoader() {
    CUFILE_CHECK(cuFileDriverOpen());
}

GDSLoader::~GDSLoader() {
    cuFileDriverClose();
}

void GDSLoader::load_weights_async(const std::string& filepath, 
                                   CUdeviceptr mapped_virtual_ptr, 
                                   size_t offset, size_t size,
                                   CUstream stream) {
#ifdef __linux__
    int fd = open(filepath.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) throw std::runtime_error("Cannot open file: " + filepath);

    CUfileDescr_t cf_desc = {};
    cf_desc.handle.fd = fd;
    cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    CUfileHandle_t cf_handle;
    CUFILE_CHECK(cuFileHandleRegister(&cf_handle, &cf_desc));
    CUFILE_CHECK(cuFileBufRegister(reinterpret_cast<void*>(mapped_virtual_ptr), size, 0));
    CUFILE_CHECK(cuFileReadAsync(cf_handle, reinterpret_cast<void*>(mapped_virtual_ptr), size, offset, offset));
    CUFILE_CHECK(cuFileBufDeregister(reinterpret_cast<void*>(mapped_virtual_ptr)));
    cuFileHandleDeregister(cf_handle);
    close(fd);
#endif
}

void GDSLoader::synchronize(CUstream stream) {
    cuStreamSynchronize(stream);
}

} // namespace hydra
