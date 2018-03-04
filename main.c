#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void fill_rand(double vet[], int size)
{
    for (int i = 0; i < size; ++i)
        vet[i] = 0.1*(rand()/(double)(RAND_MAX));
}

double*** create_network(int n_layers, int n_neurons[], int input_size)
{
    double ***network = (double***)malloc(n_layers * sizeof(double***));

    for (int i = 0; i < n_layers; ++i)
        network[i] = (double**)malloc(n_neurons[i]* sizeof(double**));

    for (int i = 0; i < n_layers; ++i)
    {
        if (i==0)
            for (int j = 0; j < n_neurons[0]; ++j)
            {
                network[0][j] = (double*)malloc(input_size * sizeof(double));
                fill_rand(network[0][j], input_size);
            }
        else
        {
            for (int j = 0; j < n_neurons[i]; ++j){
                network[i][j] = (double*)malloc(n_neurons[i-1] * sizeof(double));
                fill_rand(network[i][j], n_neurons[i-1]);
            }
        }
    }

    return network;
}


int main(void)
{
    srand(time(NULL));
    int vet[] = {1024, 10};
    printf("Started\n");
    create_network(2, vet, 28*28);
    printf("Finished");
    return 0;
}




















