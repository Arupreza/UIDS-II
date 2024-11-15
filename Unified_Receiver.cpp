#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <time.h>
#include <errno.h>
#include <sys/time.h>

#define CAN_INTERFACE "can0"
#define MAX_FILENAME_LEN 512
#define MAX_PATH_LEN 256

typedef enum {
    CAN_TYPE_CLASSIC,
    CAN_TYPE_FD
} can_type_t;

struct time_info {
    struct timeval start_time;
    struct timeval last_msg_time;
};

// 로그 파일 초기화 함수
FILE* initialize_log_file(const char* node_type, struct time_info* t_info) {
    char log_filename[MAX_FILENAME_LEN];
    char log_dir[MAX_PATH_LEN];
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    
    snprintf(log_dir, sizeof(log_dir), "receive_logs");
    mkdir(log_dir, 0755);
    
    snprintf(log_filename, sizeof(log_filename), 
             "%s/can_%s_%04d%02d%02d_%02d%02d%02d.csv",
             log_dir, node_type,
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);
    
    FILE* log_file = fopen(log_filename, "w");
    if (!log_file) {
        printf("Error creating log file: %s\n", strerror(errno));
        return NULL;
    }
    
    // 시작 시간 기록
    gettimeofday(&t_info->start_time, NULL);
    t_info->last_msg_time = t_info->start_time;

    // CSV 헤더 추가
    fprintf(log_file, "NUMBER,TIME_OFFSET,TYPE,ID,DLC,DATA_1,DATA_2,DATA_3,DATA_4,DATA_5,DATA_6,DATA_7,DATA_8,DELTA_TIME\n");
    
    printf("Logging to: %s\n", log_filename);
    return log_file;
}

// 시간 차이를 초로 계산하는 함수
double time_diff_sec(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) + 
           (end->tv_usec - start->tv_usec) / 1000000.0;
}

// 로그 메시지 작성 함수
void log_message(FILE* log_file, struct canfd_frame* frame, can_type_t type, struct time_info* t_info, int message_count) {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);

    // 시간 정보 계산
    double time_offset = time_diff_sec(&t_info->start_time, &current_time);
    double delta_t = time_diff_sec(&t_info->last_msg_time, &current_time);

    // CAN ID 처리
    int is_extended = frame->can_id & CAN_EFF_FLAG;
    unsigned int id = frame->can_id & (is_extended ? CAN_EFF_MASK : CAN_SFF_MASK);

    // CSV 형식으로 로깅
    fprintf(log_file, "%d,%.6f,Rx,%s%0*X,%d",
            message_count, time_offset,
            is_extended ? "" : "0",
            is_extended ? 8 : 3, id,
            frame->len);

    for (int i = 0; i < 8; i++) {
        if (i < frame->len) {
            fprintf(log_file, ",%02X", frame->data[i]);
        } else {
            fprintf(log_file, ",");  // 빈 필드
        }
    }

    fprintf(log_file, ",%.6f\n", delta_t);
    fflush(log_file);

    // 콘솔 출력
    printf("Message %d: Time: %.6f, %s ID: %s%0*X (%s), Data:",
           message_count, time_offset,
           type == CAN_TYPE_FD ? "CAN-FD" : "CAN",
           is_extended ? "" : "0",
           is_extended ? 8 : 3, id,
           is_extended ? "EXT" : "STD");

    for (int i = 0; i < frame->len; i++) {
        printf(" %02X", frame->data[i]);
    }
    printf(", Δt: %.6f\n", delta_t);

    t_info->last_msg_time = current_time;
}

// CAN 소켓 초기화 함수
int init_can_socket(const char* interface_name) {
    int socket_fd;
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    socket_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // CAN FD 모드 활성화
    int enable_canfd = 1;
    if (setsockopt(socket_fd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &enable_canfd, sizeof(enable_canfd))) {
        printf("Warning: CAN FD mode not supported, falling back to Classical CAN\n");
    } else {
        printf("CAN FD mode enabled\n");
    }
    
    // 모든 CAN ID 수신 허용 (Extended 포함)
    struct can_filter rfilter[1];
    rfilter[0].can_id   = 0;
    rfilter[0].can_mask = 0;
    setsockopt(socket_fd, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter));
    
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, interface_name, IFNAMSIZ - 1);
    if (ioctl(socket_fd, SIOCGIFINDEX, &ifr) < 0) {
        perror("SIOCGIFINDEX");
        close(socket_fd);
        return -1;
    }
    
    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    
    if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Bind failed");
        close(socket_fd);
        return -1;
    }
    
    return socket_fd;
}

int main(int argc, char *argv[]) {
    struct time_info t_info;
    FILE* log_file = initialize_log_file("receiver", &t_info);
    if (!log_file) {
        return 1;
    }
    
    int socket_fd = init_can_socket(CAN_INTERFACE);
    if (socket_fd < 0) {
        fclose(log_file);
        return 1;
    }
    
    printf("CAN/CANFD Receiver started. Press Ctrl+C to exit.\n");
    printf("Listening on interface: %s\n", CAN_INTERFACE);
    printf("Supporting: Standard CAN, Extended CAN, CAN FD\n");
    
    struct canfd_frame frame;
    unsigned int message_count = 0;

    while (1) {
        ssize_t nbytes = read(socket_fd, &frame, sizeof(struct canfd_frame));
        
        if (nbytes == sizeof(struct can_frame)) {
            message_count++;
            log_message(log_file, (struct canfd_frame*)&frame, CAN_TYPE_CLASSIC, &t_info, message_count);
        }
        else if (nbytes == sizeof(struct canfd_frame)) {
            message_count++;
            log_message(log_file, &frame, CAN_TYPE_FD, &t_info, message_count);
        }
        else if (nbytes == -1) {
            if (errno != EINTR) {
                perror("Read error");
            }
            break;
        }
    }
    
    printf("\nClosing CAN receiver...\n");
    close(socket_fd);
    fclose(log_file);
    return 0;
}
