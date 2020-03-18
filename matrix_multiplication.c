#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

int **allocate_2d_matrix(int m, int n);

void free_2d_matrix(int **mat);

int **mat1, **mat2, **res;
MPI_Status status;           /* return status for recieve */

int main(int argc, char *argv[]) {

    int my_rank;                 /* rank of process */
    int p;                       /* number of process */
    int source;                  /* rank of sender */
    int dest;
    int i, j, k;
    int tag = 0;                 /* tag for messages */

    int row1, col1;              /* dimensions of the first matrix */
    int row2, col2;              /* dimensions of the second matrix */
    int rows, offset, remainder, averow;    /* to divide the # of rows among processes */

    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* find out process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* find out number of process */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (my_rank == 0) {
        printf("\nWelcome to Matrix multiplication program!\n\n");

        /* Dimensions of first matrix */
        printf("Please enter dimensions of the first matrix: ");
        scanf("%d %d", &row1, &col1);

        printf("\n");

        /* Elements of the first matrix */
        printf("Please enter its elements:\n");
        mat1 = allocate_2d_matrix(row1, col1);
        for (i = 0; i < row1; i++) {
            for (j = 0; j < col1; j++) {
                scanf("%d", &mat1[i][j]);
            }
        }

        printf("\n");

        /* Dimensions of the second matrix */
        printf("Please enter dimensions of the second matrix: ");
        scanf("%d %d", &row2, &col2);

        printf("\n");

        /* Elements of the second matrix */
        printf("Please enter its elements:\n");
        mat2 = allocate_2d_matrix(row2, col2);
        for (i = 0; i < row2; i++) {
            for (j = 0; j < col2; j++) {
                scanf("%d", &mat2[i][j]);
            }
        }

        printf("\n");

        /* check if the two matrices could be multiplied */
        if (col1 == row2) {
            /* to divide work */
            res = allocate_2d_matrix(row1, col2);
            averow = row1 / (p - 1);
            remainder = row1 % (p - 1);
            offset = 0;
            tag = 1;

            for (dest = 1; dest < p; dest++) {
                /* handling remaining workload */
                rows = (dest <= remainder) ? (averow + 1) : averow;
                MPI_Send(&offset, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&col1, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&row2, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&col2, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&(mat1[offset][0]), rows * col1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(&(mat2[0][0]), row2 * col2, MPI_INT, dest, tag, MPI_COMM_WORLD);
                offset = offset + rows;
            }
            tag = 2;
            for (source = 1; source < p; source++) {
                MPI_Recv(&offset, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
                MPI_Recv(&rows, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
                MPI_Recv(&res[offset][0], rows * col2, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            }

            printf("Result Matrix is (%d x %d): \n", row1, col2);
            for (i = 0; i < row1; i++) {
                for (j = 0; j < col2; j++)
                    printf("%d ", res[i][j]);
                printf("\n");
            }

            printf("\n");

            free_2d_matrix(mat1);
            free_2d_matrix(mat2);
            free_2d_matrix(res);
        } else {
            printf("number of columns of the first matrix not equal number of rows of the second matrix\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }


    } else {
        source = 0;
        tag = 1;
        MPI_Recv(&offset, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&col1, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&row2, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&col2, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        mat1 = allocate_2d_matrix(rows, col1);
        MPI_Recv(&(mat1[0][0]), rows * col1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        mat2 = allocate_2d_matrix(row2, col2);
        MPI_Recv(&(mat2[0][0]), row2 * col2, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        res = allocate_2d_matrix(rows, col2);

        for (i = 0; i < rows; i++) {
            for (j = 0; j < col2; j++) {
                res[i][j] = 0;
                for (k = 0; k < row2; k++) {
                    res[i][j] += (mat1[i][k] * mat2[k][j]);
                }
            }
        }

        tag = 2;
        MPI_Send(&offset, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&res[0][0], rows * col2, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;

}

int **allocate_2d_matrix(int m, int n) {
    int *linear, **mat;
    int i;

    linear = malloc(sizeof(int) * m * n);
    mat = malloc(sizeof(int *) * m);
    for (i = 0; i < m; i++)
        mat[i] = &linear[i * n];

    return mat;
}

void free_2d_matrix(int **mat) {
    free(mat[0]);
    free(mat);
}

