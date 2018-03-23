#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#define activ(x) (1.0/((1.0+exp(-1.0*(x)))))
//#define deriv_activ(x) (activ(x)*(1-activ(x)))
#define activ(x) (tanh(x))
#define deriv_activ(x) (pow((2.0/(exp(x)+exp(-1.0*(x)))), 2))


struct mlp
{
    int n_layers;
    int input_size;
    double LEARNING_RATE;
    double error;
    int *n_neurons;
    double **z_outs;
    double **y_outs;
    double **deltas;
    double ***network;
};
typedef struct mlp MLP;

void fill_rand(double vet[], int size);
MLP *create_mlp(int n_layers, int n_neurons[], int input_size);
double mult(const double *x, const double *w, int size);
void forward(MLP *mlp, double *input);
void backward(MLP *mlp, double *input, double *output);
void free_mlp(MLP *mlp);

int main(void)
{
    srand(time(NULL));
    int vet[] = {2, 1};
    printf("%ld\n", sizeof(vet)/sizeof(vet[0]));
    MLP *mlp = create_mlp(sizeof(vet)/sizeof(vet[0]), vet, 2);

    double vetor1[] = {0, 0};
    double vetor2[] = {1, 0};
    double vetor3[] = {0, 1};
    double vetor4[] = {1, 1};

    double out1[] = {0};
    double out2[] = {1};

    for (int j = 0; j < 30000; ++j)
    {
        forward(mlp, vetor1);
        backward(mlp, vetor1, out1);
        forward(mlp, vetor2);
        backward(mlp, vetor2, out2);
        forward(mlp, vetor3);
        backward(mlp, vetor3, out2);
        forward(mlp, vetor4);
        backward(mlp, vetor4, out1);
        if (j%10000 == 0)
            printf("Error: %f\n", mlp->error);
    }

    double *final_outs = mlp->y_outs[mlp->n_layers-1];
    double *zs = mlp->z_outs[mlp->n_layers-1];

    printf("\n");
    forward(mlp, vetor1);
    //final_outs = mlp->y_outs[mlp->n_layers-1];
    for (int i = 0; i < mlp->n_neurons[mlp->n_layers-1]; ++i) {
        printf("%f\n", final_outs[i]);
    }

    printf("\n");
    forward(mlp, vetor2);
    //final_outs = mlp->y_outs[mlp->n_layers-1];
    for (int i = 0; i < mlp->n_neurons[mlp->n_layers-1]; ++i) {
        printf("%f\n", final_outs[i]);
    }

    printf("\n");
    forward(mlp, vetor3);
    //final_outs = mlp->y_outs[mlp->n_layers-1];
    for (int i = 0; i < mlp->n_neurons[mlp->n_layers-1]; ++i) {
        printf("%f\n", final_outs[i]);
    }

    printf("\n");
    forward(mlp, vetor4);
    //final_outs = mlp->y_outs[mlp->n_layers-1];
    for (int i = 0; i < mlp->n_neurons[mlp->n_layers-1]; ++i) {
        printf("%f\n", final_outs[i]);
    }
    backward(mlp, vetor4, out1);

    free_mlp(mlp);

    return 0;
}

void backward(MLP *mlp, double *input, double *output)
{
    double error = 0;
    int n_last_layer = mlp->n_layers-1;
    double **last_layer = mlp->network[n_last_layer]; //last layer
    double **first_layer = mlp->network[n_last_layer-1];
    for (int i = 0; i < mlp->n_neurons[n_last_layer]; ++i)//updating each weight of the layer
    {
        error += (mlp->y_outs[n_last_layer][i] - output[i])*(mlp->y_outs[n_last_layer][i] - output[i]);
        mlp->deltas[n_last_layer][i] = (mlp->y_outs[n_last_layer][i] - output[i])*
                                      deriv_activ(mlp->z_outs[n_last_layer][i]);
        for (int j = 0; j < mlp->n_neurons[n_last_layer-1]; ++j)
        {
            last_layer[i][j] -=
                    mlp->LEARNING_RATE*mlp->deltas[n_last_layer][i]
                    *(mlp->y_outs[n_last_layer-1][j]);
        }
        last_layer[i][mlp->n_neurons[n_last_layer-1]] -=
                mlp->LEARNING_RATE*mlp->deltas[n_last_layer][i];
    }

    error/=2.0;
    mlp->error = error;

    double sum;
    for (int i = 0; i < mlp->n_neurons[n_last_layer-1]; ++i)//updating each weight of the layer
    {
        sum = 0;
        for (int k = 0; k < mlp->n_neurons[n_last_layer]; ++k)
        {
            sum += mlp->deltas[n_last_layer][k]*last_layer[k][i];
        }
        for (int j = 0; j < mlp->input_size; ++j)
        {
            first_layer[i][j] -=
                    mlp->LEARNING_RATE*sum
                    *deriv_activ(mlp->z_outs[n_last_layer-1][i])
                    *(input[j]);
        }
        //bias:
        first_layer[i][mlp->input_size] -=
                mlp->LEARNING_RATE*sum
                *deriv_activ(mlp->z_outs[n_last_layer-1][i]);
    }
}

void forward(MLP *mlp, double *input)
{
    double z;
    double *w;
    for (int i = 0; i < mlp->n_neurons[0]; ++i)
    {
        w = mlp->network[0][i];
        z = mult(input, w, 2);
        mlp->z_outs[0][i] = z;
        mlp->y_outs[0][i] = activ(z);
    }
    for (int k = 1; k < mlp->n_layers; ++k)
    {
        for (int i = 0; i < mlp->n_neurons[k]; ++i)
        {
            w = mlp->network[k][i];
            z = mult(mlp->y_outs[k-1], w, mlp->n_neurons[k-1]);
            mlp->z_outs[k][i] = z;
            mlp->y_outs[k][i] = activ(z);
        }
    }
}

double mult(const double *x, const double *w, int size)
{
    double res = 0;
    for (int i = 0; i < size; ++i) {
        res += x[i]*w[i];
    }
    return res + w[size]; // w[size] = bias
}

MLP *create_mlp(int n_layers, int n_neurons[], int input_size) {

    MLP *mlp = (MLP*)malloc(sizeof(MLP));
    mlp->LEARNING_RATE = 0.05;

    double ***network = (double ***) malloc(n_layers * sizeof(double ***));
    double **z_outs = (double **) malloc(n_layers * sizeof(double *));
    double **y_outs = (double **) malloc(n_layers * sizeof(double *));
    double **deltas = (double **) malloc((n_layers) * sizeof(double *));

    for (int i = 0; i < n_layers; ++i)
    {
        network[i] = (double **) malloc(n_neurons[i] * sizeof(double **));
        z_outs[i]  = (double *)  malloc(n_neurons[i] * sizeof(double));
        y_outs[i]  = (double *)  malloc(n_neurons[i] * sizeof(double));
        deltas[i]  = (double *)  malloc(n_neurons[i] * sizeof(double));
    }

    for (int i = 0; i < n_layers; ++i)
    {
        if (i==0)
            for (int j = 0; j < n_neurons[0]; ++j)
            {
                network[0][j] = (double*)malloc((input_size+1) * sizeof(double));//+1 because of bias
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

    mlp->network = network;
    mlp->y_outs = y_outs;
    mlp->z_outs = z_outs;
    mlp->deltas = deltas;
    mlp->n_layers = n_layers;
    mlp->input_size = input_size;
    mlp->n_neurons = n_neurons;

    return mlp;
}

void fill_rand(double vet[], int size)
{
    for (int i = 0; i < size; ++i)
        vet[i] = 0.01*((rand()/(double)(RAND_MAX))*2.0-1.0);
}

void free_mlp(MLP *mlp)
{
    for (int i = 0; i < mlp->n_layers; ++i)
    {
        for (int j = 0; j < mlp->n_neurons[i]; ++j)
            free(mlp->network[i][j]);
    }

    for (int i = 0; i < mlp->n_layers; ++i)
    {
        free(mlp->network[i]);
        free(mlp->z_outs[i]);
        free(mlp->y_outs[i]);
        free(mlp->deltas[i]);
    }

    free(mlp->network);
    free(mlp->y_outs);
    free(mlp->z_outs);
    free(mlp->deltas);
    //free(mlp->n_neurons);
    free(mlp);

}
