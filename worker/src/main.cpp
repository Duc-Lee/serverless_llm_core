#include "cuda_vmm.hpp"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <chrono>

int main() {
    const size_t RESERVED_VA = 32ULL * 1024 * 1024 * 1024; // 32 GB
    const size_t CHUNK_SIZE  = 2ULL * 1024 * 1024;        // 2 MB aligned
    VirtualVramManager vmm(RESERVED_VA, CHUNK_SIZE);

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, "/tmp/hydra_worker.sock", sizeof(addr.sun_path) - 1);
    unlink(addr.sun_path);

    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);

    std::cout << "[Shadow Worker] Context initialized. Base VA: " 
              << (void*)vmm.get_base_va() << ". Listening...\n";

    while (true) {
        int client_fd = accept(server_fd, NULL, NULL);
        char buffer[128];
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            std::string cmd(buffer);

            if (cmd.rfind("HYDRATE", 0) == 0) {
                auto start = std::chrono::high_resolution_clock::now();
                // Giả lập nạp 4GB weights vào GPU VMM
                vmm.map_backing_store(4ULL * 1024 * 1024 * 1024);
                auto end = std::chrono::high_resolution_clock::now();
                
                std::cout << "[Shadow Worker] Hydrated 4GB in: "
                          << std::chrono::duration<double, std::milli>(end - start).count()
                          << " ms. Active VRAM: " << (vmm.get_active_bytes() >> 20) << " MB\n";
                write(client_fd, "ACK_READY\n", 10);
            } 
            else if (cmd.rfind("EVICT", 0) == 0) {
                vmm.evict_all();
                std::cout << "[Shadow Worker] Evicted. Active VRAM: 0 MB\n";
                write(client_fd, "ACK_ZERO\n", 9);
            }
        }
        close(client_fd);
    }
    close(server_fd);
    return 0;
}
// T?ng Shadow Worker
