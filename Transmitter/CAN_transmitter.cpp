#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>

static void signal_handler(int s) {
    printf("Interrupted by SIG%u!\n", s);
    exit(0);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <TRC file path>\n", argv[0]);
        return 1;
    }

    // Allocate memory for timestamps and CAN messages
    long *time_stemps = (long*)malloc(3000000 * sizeof(long));
    struct can_frame *messages = (struct can_frame*)malloc(3000000 * sizeof(struct can_frame));
    if (!time_stemps || !messages) {
        perror("Memory allocation failed");
        return -1;
    }

    struct timespec start, current;
    char trc_file_name[256];
    strncpy(trc_file_name, argv[1], sizeof(trc_file_name) - 1);

    FILE *input = fopen(trc_file_name, "r");
    if (!input) {
        perror("Error opening input file");
        free(time_stemps);
        free(messages);
        return 1;
    }

    // Initialize CAN socket
    int sockfd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sockfd < 0) {
        perror("Socket creation failed");
        fclose(input);
        free(time_stemps);
        free(messages);
        return -1;
    }

    struct ifreq ifr;
    strcpy(ifr.ifr_name, "can0");
    if (ioctl(sockfd, SIOCGIFINDEX, &ifr) < 0) {
        perror("Setting CAN interface failed");
        close(sockfd);
        fclose(input);
        free(time_stemps);
        free(messages);
        return -1;
    }

    struct sockaddr_can addr = {
        .can_family = AF_CAN,
        .can_ifindex = ifr.ifr_ifindex
    };

    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Binding CAN socket failed");
        close(sockfd);
        fclose(input);
        free(time_stemps);
        free(messages);
        return -1;
    }

    // Set up signal handler for graceful exit
    signal(SIGINT, signal_handler);

    // Parse TRC file
    char line[128];
    unsigned int linenumber = 0;
    for (int i = 0; i < 17; i++) fgets(line, sizeof(line), input); // Skip header lines

    char dumy_time[10];
    while (fgets(line, sizeof(line), input) && linenumber < 3000000) {
        memset(dumy_time, '\0', sizeof(dumy_time));

        // TRC 파일에서 CAN ID 및 데이터를 읽어옴
        sscanf(line, "%*d) %9s Rx %x %hhu %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx",
               dumy_time, &messages[linenumber].can_id, &messages[linenumber].can_dlc,
               &messages[linenumber].data[0], &messages[linenumber].data[1], &messages[linenumber].data[2], &messages[linenumber].data[3],
               &messages[linenumber].data[4], &messages[linenumber].data[5], &messages[linenumber].data[6], &messages[linenumber].data[7]);

        // CAN ID가 0x7FF보다 크면 확장 형식으로 설정
        if (messages[linenumber].can_id > 0x7FF) {
            messages[linenumber].can_id |= CAN_EFF_FLAG;
        }

        // time_stemps를 ns 단위로 변환하여 저장
        int len = strlen(dumy_time);
        dumy_time[len - 2] = dumy_time[len - 1];
        dumy_time[len - 1] = '\0';
        time_stemps[linenumber] = atol(dumy_time) * 100000L; // Convert to nanoseconds
        linenumber++;
    }

    printf("### File loaded successfully ###\nTotal lines processed: %u\n\n", linenumber);

    // Clear stdin buffer and wait for user input
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
    printf("Press Enter to start CAN message transmission...");
    getchar();

    // Start transmission based on timestamps
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (unsigned int i = 0; i < linenumber; i++) {
        long elapsed_time;
        long wait_time_ns = time_stemps[i] - (i > 0 ? time_stemps[i - 1] : 0);

        // Busy-wait until the specified time
        do {
            clock_gettime(CLOCK_MONOTONIC, &current);
            elapsed_time = (current.tv_sec - start.tv_sec) * 1000000000L + (current.tv_nsec - start.tv_nsec);
        } while (elapsed_time < time_stemps[i]);

        if (write(sockfd, &messages[i], sizeof(struct can_frame)) != sizeof(struct can_frame)) {
            perror("Error sending CAN message");
        }
    }

    // Clean up
    close(sockfd);
    fclose(input);
    free(time_stemps);
    free(messages);

    return 0;
}
