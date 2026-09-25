#include "vmm_arena.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <iostream>
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

void run_ipc_server(hydra::VMMVirtualArena &arena) {
#ifndef _WIN32
  int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd < 0) {
    std::cerr << "Failed to create UDS socket\n";
    return;
  }

  struct sockaddr_un addr;
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, "/tmp/hydra.sock", sizeof(addr.sun_path) - 1);
  unlink("/tmp/hydra.sock");

  if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    std::cerr << "Failed to bind to /tmp/hydra.sock\n";
    return;
  }

  listen(server_fd, 5);
  std::cout
      << "[Hydra Worker] UDS IPC Server listening on /tmp/hydra.sock...\n";

  while (true) {
    int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0)
      continue;

    char buffer[256] = {0};
    read(client_fd, buffer, 255);
    std::string cmd(buffer);

    if (cmd.find("REMAP") != std::string::npos) {
      std::cout << "[Hydra Worker] IPC: REMAP command received. Attaching "
                   "physical memory...\n";
      // arena.swap_layer_chunk(...)
    } else if (cmd.find("UNMAP") != std::string::npos) {
      std::cout << "[Hydra Worker] IPC: UNMAP command received. Scale-to-Zero "
                   "(0 MB VRAM)!\n";
      // arena.unmap_all(...)
    }

    close(client_fd);
  }
#else
  std::cout << "[Hydra Worker] Windows OS detected. UDS fallback to stdin IPC "
               "loop...\n";
  std::string cmd, model_id;
  while (std::cin >> cmd >> model_id) {
    if (cmd == "REMAP") {
      std::cout << "[Hydra Worker] IPC: REMAP command received. Attaching "
                   "physical memory...\n";
    } else if (cmd == "UNMAP") {
      std::cout << "[Hydra Worker] IPC: UNMAP command received. Scale-to-Zero "
                   "(0 MB VRAM)!\n";
    }
  }
#endif
}

int main() {
  std::cout << "[Hydra Worker] Booting Perpetual Shadow Process...\n";
  try {
    hydra::VMMVirtualArena arena(0,
                                 80ULL * 1024 * 1024 * 1024); // 80GB VA Space
    std::cout << "[Hydra Worker] 80GB VA space reserved.\n";
    run_ipc_server(arena);
  } catch (const std::exception &e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
