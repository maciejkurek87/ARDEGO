#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>

#define PI 3.141592654
#define N 1
#define MAX_LAMBDAS 1
#define MAX_MIUS 1


//Use a method described by Abramowitz and Stegun:
double gaussrand1()
{
	static double U, V;
	static int phase = 0;
	double Z;

	if(phase == 0) {
		U = (rand() + 1.) / (RAND_MAX + 2.);
		V = rand() / (RAND_MAX + 1.);
		Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
	} else
		Z = sqrt(-2 * log(U)) * cos(2 * PI * V);

	phase = 1 - phase;

	return Z;
}

//Use a method discussed in Knuth and due originally to Marsaglia:

double gaussrand2()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if(phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}

double ei_multi_min(int max_mius, double miu_mean[], double miu_s2[], int max_lambdas, double lambda_s2[], double lambda_mean[], double y_best, int n){
    double sum_ei=0.0;
    double random[100];
    double svector[100];
    for (int k=0;k<n;k++){
        double min = y_best;
        
        // prepare vector of uniform random numbers of lenght lambdas + mius
        for (int m = 0; m < max_mius + max_lambdas; m++) {
            random[m] = gaussrand1();
        }
        
        for (int l = 0; l < max_mius + max_lambdas; l++) {
            for (int ll=0; ll < max_mius + max_lambdas; ll++) {
                svector[l] += lambda_s2[ll + l  * (max_mius + max_lambdas)] * random[ll];
            }
        }
        for(int i=0;i<max_mius;i++){
            double mius = svector[i] + miu_mean[i];
            svector[i] = 0.0;
            if (mius < min)
                min = mius;
        }
        double min2=1000000000000.0;
        for(int j=0;j<max_lambdas;j++){
            double lambda = svector[j + max_mius] + lambda_mean[j];
            svector[j + max_mius] = 0.0;
            if (lambda < min2)
                min2 = lambda;
        }
        
        double e_i = min - min2;
        if (e_i < 0.0)
            e_i = 0.0;
        sum_ei = e_i + sum_ei;
    }
    return sum_ei;
}

double ei_multi_max(int max_mius, double miu_mean[], double miu_s2[], int max_lambdas, double lambda_s2[], double lambda_mean[], double y_best, int n){
    double sum_ei=0.0;
    double random[100];
    double svector[100];
    int ylenght = max_mius + max_lambdas;
    for (int k=0;k<n;k++){
        double min = y_best;
         // prepare vector of uniform random numbers of lenght lambdas + mius
        for (int m = 0; m < ylenght; m++) {
            random[m] = gaussrand1();
        }
        
        for (int l = 0; l < ylenght; l++) {
            for (int ll=0; ll < ylenght; ll++) {
                svector[l] += lambda_s2[ll + l  * ylenght] * random[ll];
            }
        }
        for(int i=0;i<max_mius;i++){
            double mius = svector[i] + miu_mean[i];
            svector[i] = 0.0;
            if (mius > min)
                min = mius;
        }
        double min2=-1000000000000.0;
        for(int j=0;j<max_lambdas;j++){
            double lambda = svector[j + max_mius] + lambda_mean[j];
            svector[j + max_mius] = 0.0;
            if (lambda > min2)
                min2 = lambda;
        }
        double e_i = min2 - min;
        if (e_i < 0.0)
            e_i = 0.0;
        sum_ei = e_i + sum_ei;
    }
    return sum_ei;
}

