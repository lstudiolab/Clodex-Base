#include "clodex/model.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* version = clodex_engine_version();
    if (version == nullptr || std::string(version).empty()) {
        std::cerr << "engine version missing\n";
        return 1;
    }

    const auto root = std::filesystem::temp_directory_path() / "clodex-engine-smoke";
    std::filesystem::create_directories(root);
    const auto corpus = root / "corpus.txt";
    const auto checkpoint = root / "smoke.ckpt";
    {
        std::ofstream out(corpus, std::ios::binary);
        out << "hello clodex. hello rust and cpp. hello clodex. ";
    }

    const int trained = clodex_train(corpus.string().c_str(), checkpoint.string().c_str(),
                                     2, 8, 16, 0.001, 0, 1);
    if (trained != 0 || !std::filesystem::is_regular_file(checkpoint)) {
        std::cerr << "training smoke test failed: " << clodex_last_error() << "\n";
        std::filesystem::remove_all(root);
        return 2;
    }

    const unsigned char prompt[] = {'h','e','l','l','o'};
    unsigned char output[8]{};
    std::size_t output_size = 0;
    const int generated = clodex_generate(checkpoint.string().c_str(), prompt, sizeof(prompt),
                                          sizeof(output), 7, 0.8, output, sizeof(output), &output_size);
    if (generated != 0 || output_size != sizeof(output)) {
        std::cerr << "generation smoke test failed: " << clodex_last_error() << "\n";
        std::filesystem::remove_all(root);
        return 3;
    }

    // Verify checkpoint loading restores optimizer/weights and that hidden sizes
    // other than the default are supported.
    const int resumed = clodex_train(corpus.string().c_str(), checkpoint.string().c_str(),
                                     1, 8, 16, 0.001, 1, 1);
    if (resumed != 0) {
        std::cerr << "resume smoke test failed: " << clodex_last_error() << "\n";
        std::filesystem::remove_all(root);
        return 4;
    }

    std::cout << version << ": train/generate/resume passed\n";
    std::filesystem::remove_all(root);
    return 0;
}
