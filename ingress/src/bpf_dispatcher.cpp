#include <bpf/libbpf.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <iostream>

struct early_signal_t {
    uint32_t model_id;
    uint32_t payload_len;
};

static int handle_signal(void *ctx, void *data, size_t len) {
    auto *sig = static_cast<early_signal_t*>(data);
    int uds_fd = *static_cast<int*>(ctx);

    // Forward tín hiệu qua Unix Domain Socket thẳng sang Worker Engine
    char cmd[32];
    int n = snprintf(cmd, sizeof(cmd), "HYDRATE:%u\n", sig->model_id);
    write(uds_fd, cmd, n);
    return 0;
}

int main() {
    // Kết nối tới worker socket
    int uds_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, "/tmp/hydra_worker.sock", sizeof(addr.sun_path) - 1);
    
    while (connect(uds_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cout << "[Ingress] Waiting for Worker UDS socket...\n";
        sleep(1);
    }

    struct bpf_object *obj = bpf_object__open("fast_peek.bpf.o");
    bpf_object__load(obj);
    
    struct ring_buffer *rb = ring_buffer__new(
        bpf_object__find_map_fd_by_name(obj, "ringbuf"),
        handle_signal, &uds_fd, NULL
    );

    std::cout << "[Ingress] eBPF Ingress Monitor Active.\n";
    while (true) {
        ring_buffer__poll(rb, 100);
    }

    ring_buffer__free(rb);
    bpf_object__close(obj);
    close(uds_fd);
    return 0;
}
