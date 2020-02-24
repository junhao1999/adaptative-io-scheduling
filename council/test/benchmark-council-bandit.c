/* File:    benchmark-council-bandit.c
 * Created: 2019
 * License: GPL version 3
 * Author:
 *      Jean Luca Bez <jean.bez (at) inf.ufrgs.br>
 *
 * Description:
 *      Benchmark the Council by sending fake metrics and receiving new instructions
 *
 * Contributors:
 *      Federal University of Rio Grande do Sul (UFRGS)
 *
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <time.h>
#include <unistd.h>

#define OBSERVATIONS 60

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[])
{
    int i, sockfd, portno, n, force_twins;
    float elapsed, total = 0.0;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    struct timespec ts1, ts2, tsr;

    srand((unsigned int)time(NULL));

    char buffer[256];
    char window_buffer[8];

    if (argc < 3) {
       fprintf(stderr, "usage %s hostname port \n", argv[0]);
       exit(0);
    }

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
    if (sockfd < 0) {
        error("ERROR opening socket");
    }
    
    server = gethostbyname(argv[1]);
    
    if (server == NULL) {
        fprintf(stderr, "ERROR, no such host\n");
        exit(0);
    }
    
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr,
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);

    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) {
        error("ERROR connecting");
    }

    // Example of metrics in the CSV file:
    sprintf(buffer, "%d;%d;%d;%d;%d;%.10lf;%d;%d;%.10f;%.10f\n",
        4,
        32,
        128,
        1,
        1,32.0,32,32,18769.83984375,
        10.0
    );

    // Receives OK message
    n = recv(sockfd, window_buffer, 8, MSG_WAITALL);

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < OBSERVATIONS; i++) {

    	clock_gettime(CLOCK_MONOTONIC, &ts1);

        // Send the fake metrics to the Council
        n = write(sockfd, buffer, strlen(buffer));
        if (n < 0) {
            error("ERROR writing to socket");
        }

        // Receive the new window size
        n = recv(sockfd, window_buffer, 8, MSG_WAITALL);

        clock_gettime(CLOCK_MONOTONIC, &tsr);
        
        if (n < 0) {
            error("ERROR reading from socket");
        }

        clock_gettime(CLOCK_MONOTONIC, &ts2);

        if (ts2.tv_nsec < ts1.tv_nsec) {
    		ts2.tv_nsec += 1000000000;
    		ts2.tv_sec--;
    	}

        elapsed = (long)(ts2.tv_sec - ts1.tv_sec) + ((ts2.tv_nsec - ts1.tv_nsec) / 1000000000.0);

    	printf("%d;%d;%.9lf;%d\n", world_rank, i, elapsed * 1000, atoi(window_buffer));

        sleep(1);
    }

    close(sockfd);

    MPI_Finalize();

    return 0;
}
