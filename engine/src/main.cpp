#include "hydra_engine.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <thread>

void stdin_ipc_listener(hydra::HydraEngine& engine) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        uint32_t model_hash;
        size_t tokens;
        
        if (iss >> model_hash >> tokens) {
            engine.trigger_inference(model_hash, tokens);
        } else {
            std::cerr << "Daemon: Invalid IPC command format.\n";
        }
    }
}

int main() {
    try {
        hydra::HydraEngine engine(0, 16ULL * 1024 * 1024 * 1024);

        engine.register_tenant(2166136261U, "/opt/models/llama-3-8b.bin", 2ULL * 1024 * 1024 * 1024); 
        engine.register_tenant(999999999U, "/opt/models/mistral-7b.bin", 2ULL * 1024 * 1024 * 1024);

        std::thread listener(stdin_ipc_listener, std::ref(engine));
        listener.join();

    } catch (const std::exception& e) {
        std::cerr << "Daemon Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
