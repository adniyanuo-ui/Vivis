// Read-only cross-check using the unmodified official FTServo_Linux sources.
#include "SCServo.h"
#include <chrono>
#include <cstdio>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    SMS_STS servo;
    if (!servo.begin(1000000, argv[1])) return 3;
    const auto start = std::chrono::steady_clock::now();
    for (int sample = 0; sample < 150; ++sample) {
        const int id = 2 + sample % 5;
        unsigned char data[15] = {};
        const int size = servo.Read(id, 56, data, sizeof data);
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("{\"elapsed_us\":%lld,\"id\":%d,\"size\":%d,\"communication_error\":%d,\"servo_status\":%d,\"raw\":\"",
                    static_cast<long long>(us), id, size, servo.getLastError(), servo.getState());
        if (size == 15) for (unsigned char byte : data) std::printf("%02x", byte);
        std::printf("\"}\n");
        usleep(10000);
    }
    // Process exit closes the serial fd; avoid SDK end()'s fd=-1-before-close bug.
    return 0;
}
