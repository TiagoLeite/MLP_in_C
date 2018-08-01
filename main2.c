#include <stdlib.h>
#include <stdio.h>

unsigned char** read_data(int n, int width, int height);

int main(void)
{
    /*for (int i = 0; i < 16; ++i) {
        fread(&num, sizeof(char), 1, f);
        printf("%d\n", num);
    }*/

    unsigned char** data = read_data(10, 28, 28);
    free(data[0]);
    free(data);

}

unsigned char** read_data(int n, int width, int height)
{
    unsigned char **vet;
    vet = (unsigned char**)malloc(n*sizeof(unsigned char*));
    vet[0] = (unsigned char*)malloc(n*height*width*sizeof(unsigned char));

    for (int k = 1; k < n; ++k)
        vet[k] = vet[0] + k * width * height;

    FILE *f = fopen("train-images.idx3-ubyte", "rb");
    unsigned char num;
    fseek(f, 16, SEEK_CUR);
    int read = (int)fread(*vet, sizeof(unsigned char), (unsigned int)(n * width * height), f);
    printf("Read %d bytes\n", read);
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < width * height; i++)
        {
            num = vet[j][i];
            if (num > 0)
                printf("1");
            else
                printf("0");
            if (i % 28 == 27)
                printf("\n");
        }
        printf("=============\n");
    }
    fclose(f);
    return vet;
}
