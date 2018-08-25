#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <memory.h>
//#define activ(x) (1.0/((1.0+exp(-1.0*(x)))))
//#define deriv_activ(x) (activ(x)*(1-activ(x)))
//#define activ(x) (tanh(x))
//#define deriv_activ(x) (pow((2.0/(exp(x)+exp(-1.0*(x)))), 2))
#define activ(x) ((x) > 0 ? (x) : 0.0)
#define deriv_activ(x) ((x) > 0 ? 1.0 : 0.0)

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
void forward(MLP *mlp, const double *input);
void backward(MLP *mlp, const double *input, const double *output);
void free_mlp(MLP *mlp);
double** read_data(int n, int width, int height, char file_name[], int seek);
double** read_labels(int n, char file_name[], int seek);
double* one_hot(double index, unsigned int size);
int argmax(const double* arr, int size);
double** forward_batch(MLP *mlp, double **input, int input_size);
double count_hits(double **arr1, double **arr2, int rows, int columns);
void set_error(MLP *mlp, double *output);

int main(void)
{
    srand((unsigned int) time(NULL));
    int vet[] = {10, 10};
    printf("%ld\n", sizeof(vet)/sizeof(vet[0]));
    MLP *mlp = create_mlp(sizeof(vet)/sizeof(vet[0]), vet, 28*28);

    double **train_labels = read_labels(50000, "train-labels.idx1-ubyte", 8);
    double **train_images = read_data(50000, 28, 28, "train-images.idx3-ubyte", 16);

    double **test_labels = read_labels(10000, "t10k-labels.idx1-ubyte", 8);
    double **test_images = read_data(10000, 28, 28, "t10k-images.idx3-ubyte", 16);

    int cont = 0;
    while (cont < 10)
    {
        printf("Epoca %d\n", cont);
        for (int i = 0; i < 50000; ++i)
        {
            forward(mlp, train_images[i]);
            backward(mlp, train_images[i], train_labels[i]);
            set_error(mlp, train_labels[i]);
            if (i % 5000 == 0)
                printf("Error: %.3f\n", mlp->error);
            //printf("%f %f %f %f\n", mlp->deltas[0][5], mlp->deltas[0][2],
            //       mlp->deltas[0][4], mlp->deltas[0][2]);
        }
        cont++;
    }
    printf("%d iterations", cont);
    double **out = forward_batch(mlp, test_images, 10000);
    double hits = count_hits(out, test_labels, 10000, 10);
    printf("\nAccuracy: %f\n", hits);

    free_mlp(mlp);
    return 0;
}

void backward(MLP *mlp, const double *input, const double *output)
{
    int n_last_layer = mlp->n_layers-1;
    double **last_layer = mlp->network[n_last_layer]; //last layer
    double **first_layer = mlp->network[n_last_layer-1];
    for (int i = 0; i < mlp->n_neurons[n_last_layer]; ++i)//updating each weight of the layer
    {
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

    double sum;
    for (int i = 0; i < mlp->n_neurons[n_last_layer-1]; ++i)//updating each weight of the layer
    {
        sum = 0;
        for (int k = 0; k < mlp->n_neurons[n_last_layer]; ++k)
        {
            sum += mlp->deltas[n_last_layer][k]*last_layer[k][i];
        }

        mlp->deltas[n_last_layer-1][i] = sum*deriv_activ(mlp->z_outs[n_last_layer-1][i]);

        for (int j = 0; j < mlp->input_size; ++j)
        {
            first_layer[i][j] -=
                    mlp->LEARNING_RATE*mlp->deltas[n_last_layer-1][i]
                    *(input[j]);
        }
        //bias:
        first_layer[i][mlp->input_size] -=
                mlp->LEARNING_RATE*sum
                *deriv_activ(mlp->z_outs[n_last_layer-1][i]);
    }
}

void forward(MLP *mlp, const double *input)
{
    double z, *w;

    for (int i = 0; i < mlp->n_neurons[0]; ++i)
    {
        w = mlp->network[0][i];
        z = mult(input, w, 28*28);
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

void set_error(MLP *mlp, double *output)
{
    double error = 0.0;
    int n_last_layer = mlp->n_layers-1;
    for (int i = 0; i < mlp->n_neurons[n_last_layer]; ++i)
    {
        error += (mlp->y_outs[n_last_layer][i] - output[i])*(mlp->y_outs[n_last_layer][i] - output[i]);
    }
    mlp->error = error/2.0;
}

double** forward_batch(MLP *mlp, double **input, int input_size)
{
    double z;
    double *w;
    double **outs = (double**)malloc(input_size*sizeof(double*));

    for (int j = 0; j < input_size; ++j)
    {
        outs[j] = (double*)malloc(sizeof(double)*(mlp->n_neurons[mlp->n_layers-1]));
        for (int i = 0; i < mlp->n_neurons[0]; ++i)
        {
            w = mlp->network[0][i];
            z = mult(input[j], w, 28*28);
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
        memcpy(outs[j], mlp->y_outs[mlp->n_layers-1], sizeof(double)*(mlp->n_neurons[mlp->n_layers-1]));
    }
    return outs;
}


double mult(const double *x, const double *w, int size)
{
    double res = 0.0;
    for (int i = 0; i < size; ++i) {
        res += (x[i])*w[i];
    }
    return res + w[size]; // w[size] = bias
}

MLP *create_mlp(int n_layers, int n_neurons[], int input_size) {

    MLP *mlp = (MLP*)malloc(sizeof(MLP));
    mlp->LEARNING_RATE = 5*1e-4;
    mlp->error = 1.0;

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

double** read_data(int n, int width, int height, char file_name[], int seek)
{
    unsigned char **vet;
    vet = (unsigned char**)malloc(n*sizeof(unsigned char*));
    vet[0] = (unsigned char*)malloc(n*height*width*sizeof(unsigned char));

    double **vet2;
    vet2 = (double**)malloc(n*sizeof(double*));
    vet2[0] = (double*)malloc(n*height*width*sizeof(double));

    for (int k = 1; k < n; ++k)
        vet[k] = vet[0] + k * width * height;

    for (int k = 1; k < n; ++k)
        vet2[k] = vet2[0] + k * width * height;

    FILE *f = fopen(file_name, "rb");

    fseek(f, seek, SEEK_CUR);

    int read = (int)fread(*vet, sizeof(unsigned char), (unsigned int)(n * width * height), f);

    printf("Read %d bytes\n", read);

    fclose(f);

    /*for (int i = 0; i < 784; ++i) {
        printf("%3d ", vet[3][i]);
        if (i % 28 == 27)
            printf("\n");
    }
    getchar();*/

    for (int j = 0; j < n*height*width; ++j)
    {
        vet2[0][j] = ((double)(vet[0][j]))/255.0;
        //printf("%f\n", vet2[0][j]);
    }

    return vet2;
}

double** read_labels(int n, char file_name[], int seek)
{
    unsigned char *vet;
    vet = (unsigned char *) malloc(n * sizeof(unsigned char));

    double **vet2;
    vet2 = (double**) malloc(n * sizeof(double*));

    FILE *f = fopen(file_name, "rb");
    fseek(f, seek, SEEK_CUR);
    fread(vet, sizeof(unsigned char), (unsigned int) (n), f);

    //printf("%d %d", vet[2], vet[3]);

    fclose(f);

    for (int i = 0; i < n; ++i)
        vet2[i] = one_hot(vet[i], 10);

    free(vet);

    return vet2;
}

double* one_hot(double index, unsigned int size)
{
    double* arr = (double*)calloc(sizeof(double), size);
    arr[(int)index] = 1.0;
    return arr;
}

int argmax(const double* arr, int size)
{
    if (size == 0)
        return -1;
    double max = arr[0];
    int index = 0;
    for (int i = 1; i < size; ++i)
    {
        if (arr[i] > max)
        {
            max = arr[i];
            index = i;
        }
    }
    return index;
}

double count_hits(double **arr1, double **arr2, int rows, int columns)
{
    int hits = 0;
    for (int i = 0; i < rows; ++i)
    {
        if(argmax(arr1[i], columns) == argmax(arr2[i], columns))
            hits++;
    }
    return ((double)hits)/(double)rows;
}


