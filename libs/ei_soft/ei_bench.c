#include <time.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <stdio.h>
#include "ei_multi.h"

int main(int argc, char *argv[]) {
    int n;
    int max_lambdas;
    int max_mius;
    double y_best;
    sscanf(argv[1],"%d",&n);
    sscanf(argv[2],"%d",&max_lambdas);
    sscanf(argv[2],"%d",&max_mius);
    sscanf(argv[3],"%f",&y_best);
    
    clock_t begin, end;
    double time_spent;
    double miu_mean[20];
    double miu_s2[20];
    double lambda_s2[20];
    double lambda_mean[20];

    for(int i=0;i<=20;i++){
     miu_mean[i]=gaussrand1();
     miu_s2[i]=gaussrand1();
     lambda_s2[i]=gaussrand1();
     lambda_mean[i]=gaussrand1();
    }
    //printf("Preparing benchmark for nsims: %d \n", n);
    begin = clock();
    e_multi(1, miu_mean, miu_s2, max_lambdas, lambda_s2, lambda_mean, y_best, n);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%d:%f,",n,time_spent);
    
}


