#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

struct mlp
{
    int n_layers;
    int input_size;
    int *n_neurons;
    double ***network;
};
typedef struct mlp MLP;

void fill_rand(double vet[], int size);
MLP create_mlp(int n_layers, int n_neurons[], int input_size);

int main(void)
{
    srand(time(NULL));
    int vet[] = {1024, 10};
    printf("Started\n");
    MLP mlp = create_mlp(2, vet, 28*28);
    printf("Finished");
    printf("\n%d %d", mlp.input_size, mlp.n_layers);
    return 0;
}

MLP create_mlp(int n_layers, int n_neurons[], int input_size)
{
    MLP mlp;

    double ***network = (double***)malloc(n_layers * sizeof(double***));

    for (int i = 0; i < n_layers; ++i)
        network[i] = (double**)malloc(n_neurons[i]* sizeof(double**));

    for (int i = 0; i < n_layers; ++i)
    {
        if (i==0)
            for (int j = 0; j < n_neurons[0]; ++j)
            {
                network[0][j] = (double*)malloc((input_size+3) * sizeof(double));//+3 because of bias, z and y
                fill_rand(network[0][j], input_size);
            }
        else
        {
            for (int j = 0; j < n_neurons[i]; ++j){
                network[i][j] = (double*)malloc((n_neurons[i-1]+3) * sizeof(double));
                fill_rand(network[i][j], n_neurons[i-1]);
            }
        }
    }

    mlp.network = network;
    mlp.n_layers = n_layers;
    mlp.input_size = input_size;
    mlp.n_neurons = n_neurons;

    return mlp;
}


void fill_rand(double vet[], int size)
{
    for (int i = 0; i < size; ++i)
        vet[i] = 0.1*(rand()/(double)(RAND_MAX));
}















