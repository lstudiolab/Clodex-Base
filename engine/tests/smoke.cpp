#include "clodex/model.h"
#include <cstring>
#include <iostream>

int main() {
    const char* version = clodex_engine_version();
    if (version == nullptr || std::strlen(version) == 0) {
        std::cerr << "engine version missing\n";
        return 1;
    }
    std::cout << version << "\n";
    return 0;
}
