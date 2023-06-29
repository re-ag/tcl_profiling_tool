#include <sys/timerfd.h>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <inttypes.h> /* Definition of PRIu64 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h> /* Definition of uint64_t */

#define handle_error(msg)   \
    do                      \
    {                       \
        perror(msg);        \
        exit(EXIT_FAILURE); \
    } while (0)

int main(int argc, char *argv[])
{
    struct itimerspec new_value;
    int max_exp, fd;
    struct timespec now;
    uint64_t exp, tot_exp;
    ssize_t s;
    tot_exp = 0;

    if (clock_gettime(CLOCK_REALTIME, &now) == -1)
        handle_error("clock_gettime");

    /* Create a CLOCK_REALTIME absolute timer with initial
       expiration and interval as specified in command line. */

    new_value.it_value.tv_sec = now.tv_sec;
    new_value.it_value.tv_nsec = now.tv_nsec;

    new_value.it_interval.tv_sec = 4;
    new_value.it_interval.tv_nsec = 0;

    fd = timerfd_create(CLOCK_REALTIME, 0);
    if (fd == -1)
        handle_error("timerfd_create");

    if (timerfd_settime(fd, TFD_TIMER_ABSTIME, &new_value, NULL) == -1)
        handle_error("timerfd_settime");

    printf("timer started\n");

    bool tmp = true;
    while (1)
    {
        struct timespec start;
        clock_gettime(CLOCK_REALTIME, &start);
        std::cout << "start: " << start.tv_sec << std::endl;
        printf("read start\n\n");

        s = read(fd, &exp, sizeof(uint64_t));
        
        struct timespec end;
        clock_gettime(CLOCK_REALTIME, &end);
        std::cout << "end: " << end.tv_sec << std::endl;
        printf("read end\n\n");

        if (s != sizeof(uint64_t))
            handle_error("read");

        tot_exp += exp;
        printf("read: %" PRIu64 "; total=%" PRIu64 "\n", exp, tot_exp);
        if (tmp)
        {
            sleep(13);
            tmp = false;
        }
    }

    exit(EXIT_SUCCESS);
}
