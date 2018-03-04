#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define activ(x) (1.0/(1.0 + exp(x)))
#define deriv_activ(x) (activ(x)*(1.0-activ(x)))

struct mlp
{
    int n_layers;
    int input_size;
    int *n_neurons;
    double **z_outs;
    double **y_outs;
    double ***network;
};
typedef struct mlp MLP;

void fill_rand(double vet[], int size);
MLP create_mlp(int n_layers, int n_neurons[], int input_size);
double mult(double x[], double w[], int size);
void forward(MLP mlp, double *input);

int main(void)
{
    srand(time(NULL));
    int vet[] = {2, 5};
    printf("Started\n");
    printf("%d\n", sizeof(vet)/sizeof(vet[0]));
    MLP mlp = create_mlp(sizeof(vet)/sizeof(vet[0]), vet, 10);
    printf("Finished\n");
    double vetor[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    forward(mlp, vetor);

    double *final_outs = mlp.y_outs[mlp.n_layers-1];

    for (int i = 0; i < mlp.n_neurons[mlp.n_layers-1]; ++i) {
       printf("%f", final_outs[i]);
    }


    return 0;
}

void forward(MLP mlp, double *input)
{
    double z;
    double **layer = mlp.network[0];
    for (int i = 0; i < mlp.n_neurons[0]; ++i)
    {
        z = mult(input, layer[i], mlp.input_size);
        mlp.z_outs[0][i] = z; //z and y are the last two values
        mlp.y_outs[0][i] = activ(z);
    }
    for (int k = 1; k < mlp.n_layers; ++k)
    {
        layer = mlp.network[k];
        for (int i = 0; i < mlp.n_neurons[k]; ++i)
        {
            z = mult(mlp.y_outs[k-1], layer[i], mlp.n_neurons[k-1]);
            mlp.z_outs[k][i] = z;
            mlp.y_outs[k][i] = activ(z);
        }
    }
}

double mult(double x[], double w[], int size)
{
    double res = 0;
    for (int i = 0; i < size; ++i) {
        res += x[i]*w[i];
    }
    return (res + w[size]); // w[size] = bias
}

MLP create_mlp(int n_layers, int n_neurons[], int input_size) {
    MLP mlp;

    double ***network = (double ***) malloc(n_layers * sizeof(double ***));
    double **z_outs = (double **) malloc(n_layers * sizeof(double *));
    double **y_outs = (double **) malloc(n_layers * sizeof(double *));

    for (int i = 0; i < n_layers; ++i)
    {
        network[i] = (double **) malloc(n_neurons[i] * sizeof(double **));
        z_outs[i]  = (double *)  malloc(n_neurons[i] * sizeof(double));
        y_outs[i]  = (double *)  malloc(n_neurons[i] * sizeof(double));
    }

    for (int i = 0; i < n_layers; ++i)
    {
        if (i==0)
            for (int j = 0; j < n_neurons[0]; ++j)
            {
                network[0][j] = (double*)malloc((input_size+1) * sizeof(double));//+1 because of bias, z and y
                fill_rand(network[0][j], input_size+1);
            }
        else
        {
            for (int j = 0; j < n_neurons[i]; ++j){
                network[i][j] = (double*)malloc((n_neurons[i-1]+1) * sizeof(double));
                fill_rand(network[i][j], n_neurons[i-1]+1);
            }
        }
    }

    mlp.network = network;
    mlp.y_outs = y_outs;
    mlp.z_outs = z_outs;
    mlp.n_layers = n_layers;
    mlp.input_size = input_size;
    mlp.n_neurons = n_neurons;

    return mlp;
}

void free_mlp(MLP mlp)
{
    //TODO: free memory
}


void fill_rand(double vet[], int size)
{
    for (int i = 0; i < size; ++i)
        vet[i] = 0.1*(rand()/(double)(RAND_MAX));
}

