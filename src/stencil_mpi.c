#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

void print_cube(int n, double *u) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                printf("%3.2f ", u[(i * n + j) * n + k]);
            }
            printf("\n");
        }
        printf("upper level\n");
    }
}

void print_csv(int n, double *u) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                printf("%.16f,", u[(i * n + j) * n + k]);
            }
        }
    }
}

void print_square(int n, double *u) {
    int j, k;
    for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
            printf("%3.2f ", u[j * n + k]);
        }
        printf("\n");
    }
}

/* one time step of computations */
void timeStep(int n, double *u0, double *u1, double r)
{
    int i, j, k;

    for (i = 1; i < n - 1; i++)
        for (j = 1; j < n - 1; j++)
            for (k = 1; k < n - 1; k++)
                u1[(i * n + j) * n + k] = (1.0 - 6.0 * r) * u0[(i * n + j) * n + k] + r * (u0[((i + 1) * n + j) * n + k] + u0[((i - 1) * n + j) * n + k] + u0[(i * n + j + 1) * n + k] + u0[(i * n + j - 1) * n + k] + u0[(i * n + j) * n + k + 1] + u0[(i * n + j) * n + k - 1]);
}

void solveMPI(int n, double *u0, double r, int nt, int argc, char **argv)
{
    int i, j, k;
    // Initializing MPI
    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Status stat;
    MPI_Request req;

    int sub_n = (int) cbrt(pow(n-2,3)/nproc);
    double *sub_u0 = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2) * (sub_n+2));
    double *sub_u1 = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2) * (sub_n+2));
    // Coordinates converting
    int i0 = sub_n * (int)(rank*pow((float)sub_n/(n-2),2));
    int j0 = sub_n * (int)((rank - (int)(i0*((float)pow(n-2,2)/(float)pow(sub_n,3)))) / ((n-2)/sub_n));
    int k0 = (int)(rank * sub_n % (n-2));
    // Filling u0 and u1 of the subcube with the corresponding values
    for (i = 0; i < sub_n+2; i++) {
        for (j = 0; j < sub_n+2; j++) {
            for (k = 0; k < sub_n+2; k++) {
                sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k] = u0[((i0+i) * n + (j0+j)) * n + (k0+k)];
                sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k] = u0[((i0+i) * n + (j0+j)) * n + (k0+k)];
            }
        }
    }

    // Initialization of boundary sharing arrays
    bool i_pos = (i0 < (n-2) - sub_n);
    bool i_neg = (i0 > 0);
    bool j_pos = (j0 < (n-2) - sub_n);
    bool j_neg = (j0 > 0);
    bool k_pos = (k0 < (n-2) - sub_n);
    bool k_neg = (k0 > 0);

    double *i_pos_send = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    double *i_pos_recv = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    int i_pos_rank = rank + pow((n-2)/sub_n,2);
    double *i_neg_send = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    double *i_neg_recv = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    int i_neg_rank = rank - pow((n-2)/sub_n,2);
    double *j_pos_send = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    double *j_pos_recv = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    int j_pos_rank = rank + (n-2)/sub_n;
    double *j_neg_send = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    double *j_neg_recv = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    int j_neg_rank = rank - (n-2)/sub_n;
    double *k_pos_send = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    double *k_pos_recv = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    int k_pos_rank = rank + 1;
    double *k_neg_send = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    double *k_neg_recv = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2));
    int k_neg_rank = rank - 1;

    int t;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (t = 0; t < nt; t += 2)
    {
        //printf("rank %d iteration %d\n", rank, t);
        //
        /* TIME STEPS 1, 3, 5, ... */
        //
        timeStep(sub_n+2, sub_u0, sub_u1, r);
        MPI_Barrier(MPI_COMM_WORLD);
        if (i_pos)  {
            i = sub_n;
            for (j = 0; j < sub_n+2; j++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    i_pos_send[j * (sub_n+2) + k] = sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(i_pos_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, i_pos_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(i_pos_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, i_pos_rank, 0, MPI_COMM_WORLD, &stat);
            i = sub_n + 1;
            for (j = 0; j < sub_n+2; j++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k] = i_pos_recv[j * (sub_n+2) + k];
                }
            }
        }
        if (i_neg)  {
            i = 1;
            for (j = 0; j < sub_n+2; j++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    i_neg_send[j * (sub_n+2) + k] = sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(i_neg_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, i_neg_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(i_neg_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, i_neg_rank, 0, MPI_COMM_WORLD, &stat);
            i = 0;
            for (j = 0; j < sub_n+2; j++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k] = i_neg_recv[j * (sub_n+2) + k];
                }
            }
        }
        if (j_pos)  {
            j = sub_n;
            for (i = 0; i < sub_n+2; i++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    j_pos_send[i * (sub_n+2) + k] = sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(j_pos_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, j_pos_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(j_pos_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, j_pos_rank, 0, MPI_COMM_WORLD, &stat);
            j = sub_n + 1;
            for (i = 0; i < sub_n+2; i++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k] = j_pos_recv[i * (sub_n+2) + k];
                }
            }
        }
        if (j_neg)  {
            j = 1;
            for (i = 0; i < sub_n+2; i++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    j_neg_send[i * (sub_n+2) + k] = sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(j_neg_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, j_neg_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(j_neg_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, j_neg_rank, 0, MPI_COMM_WORLD, &stat);
            j = 0;
            for (i = 0; i < sub_n+2; i++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k] = j_neg_recv[i * (sub_n+2) + k];
                }
            }
        }
        if (k_pos)  {
            k = sub_n;
            for (i = 0; i < sub_n+2; i++)  {
                for (j = 0; j < sub_n+2; j++)  {
                    k_pos_send[i * (sub_n+2) + j] = sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(k_pos_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, k_pos_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(k_pos_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, k_pos_rank, 0, MPI_COMM_WORLD, &stat);
            k = sub_n + 1;
            for (i = 0; i < sub_n+2; i++)  {
                for (j = 0; j < sub_n+2; j++)  {
                    sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k] = k_pos_recv[i * (sub_n+2) + j];
                }
            }
        }
        if (k_neg)  {
            k = 1;
            for (i = 0; i < sub_n+2; i++)  {
                for (j = 0; j < sub_n+2; j++)  {
                    k_neg_send[i * (sub_n+2) + j] = sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(k_neg_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, k_neg_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(k_neg_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, k_neg_rank, 0, MPI_COMM_WORLD, &stat);
            k = 0;
            for (i = 0; i < sub_n+2; i++)  {
                for (j = 0; j < sub_n+2; j++)  {
                    sub_u1[(i * (sub_n+2) + j) * (sub_n+2) + k] = k_neg_recv[i * (sub_n+2) + j];
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        //printf("rank %d iteration %d\n", rank, t+1);
        //
        /* TIME STEPS 2, 4, 6, ... */
        //
        timeStep(sub_n+2, sub_u1, sub_u0, r);
        MPI_Barrier(MPI_COMM_WORLD);
        if (i_pos)  {
            i = sub_n;
            for (j = 0; j < sub_n+2; j++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    i_pos_send[j * (sub_n+2) + k] = sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(i_pos_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, i_pos_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(i_pos_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, i_pos_rank, 0, MPI_COMM_WORLD, &stat);
            i = sub_n + 1;
            for (j = 0; j < sub_n+2; j++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k] = i_pos_recv[j * (sub_n+2) + k];
                }
            }
        }
        if (i_neg)  {
            i = 1;
            for (j = 0; j < sub_n+2; j++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    i_neg_send[j * (sub_n+2) + k] = sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(i_neg_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, i_neg_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(i_neg_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, i_neg_rank, 0, MPI_COMM_WORLD, &stat);
            i = 0;
            for (j = 0; j < sub_n+2; j++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k] = i_neg_recv[j * (sub_n+2) + k];
                }
            }
        }
        if (j_pos)  {
            j = sub_n;
            for (i = 0; i < sub_n+2; i++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    j_pos_send[i * (sub_n+2) + k] = sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(j_pos_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, j_pos_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(j_pos_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, j_pos_rank, 0, MPI_COMM_WORLD, &stat);
            j = sub_n + 1;
            for (i = 0; i < sub_n+2; i++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k] = j_pos_recv[i * (sub_n+2) + k];
                }
            }
        }
        if (j_neg)  {
            j = 1;
            for (i = 0; i < sub_n+2; i++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    j_neg_send[i * (sub_n+2) + k] = sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(j_neg_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, j_neg_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(j_neg_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, j_neg_rank, 0, MPI_COMM_WORLD, &stat);
            j = 0;
            for (i = 0; i < sub_n+2; i++)  {
                for (k = 0; k < sub_n+2; k++)  {
                    sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k] = j_neg_recv[i * (sub_n+2) + k];
                }
            }
        }
        if (k_pos)  {
            k = sub_n;
            for (i = 0; i < sub_n+2; i++)  {
                for (j = 0; j < sub_n+2; j++)  {
                    k_pos_send[i * (sub_n+2) + j] = sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(k_pos_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, k_pos_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(k_pos_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, k_pos_rank, 0, MPI_COMM_WORLD, &stat);
            k = sub_n + 1;
            for (i = 0; i < sub_n+2; i++)  {
                for (j = 0; j < sub_n+2; j++)  {
                    sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k] = k_pos_recv[i * (sub_n+2) + j];
                }
            }
        }
        if (k_neg)  {
            k = 1;
            for (i = 0; i < sub_n+2; i++)  {
                for (j = 0; j < sub_n+2; j++)  {
                    k_neg_send[i * (sub_n+2) + j] = sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k];
                }
            }
            MPI_Isend(k_neg_send, (sub_n+2)*(sub_n+2), MPI_DOUBLE, k_neg_rank, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(k_neg_recv, (sub_n+2)*(sub_n+2), MPI_DOUBLE, k_neg_rank, 0, MPI_COMM_WORLD, &stat);
            k = 0;
            for (i = 0; i < sub_n+2; i++)  {
                for (j = 0; j < sub_n+2; j++)  {
                    sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k] = k_neg_recv[i * (sub_n+2) + j];
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        //print_cube(sub_n+2,sub_u0);
        printf("n = %d ; nproc = %d\n", n, nproc);
        printf("time = %f\n", end-start);
    }

    //Reconstruct the cube
    if (rank == 0) {
        double *mpi_u = (double *)malloc(sizeof(double) * n * n * n);
        double *recv_u = (double *)malloc(sizeof(double) * (sub_n+2) * (sub_n+2) * (sub_n+2));
        for (i = 0; i < sub_n+2; i++)
            for (j = 0; j < sub_n+2; j++)
                for (k = 0; k < sub_n+2; k++)
                    mpi_u[((i0+i) * n + (j0+j)) * n + (k0+k)] = sub_u0[(i * (sub_n+2) + j) * (sub_n+2) + k];
        for (int sender = 1; sender < nproc; sender++) {
            MPI_Recv(&i0, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, &stat);
            MPI_Recv(&j0, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, &stat);
            MPI_Recv(&k0, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, &stat);
            MPI_Recv(recv_u, (sub_n+2)*(sub_n+2)*(sub_n+2), MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &stat);
            for (i = 0; i < sub_n+2; i++)
                for (j = 0; j < sub_n+2; j++)
                    for (k = 0; k < sub_n+2; k++)
                        mpi_u[((i0+i) * n + (j0+j)) * n + (k0+k)] = recv_u[(i * (sub_n+2) + j) * (sub_n+2) + k];
        }
        print_csv(n, mpi_u);
        //print_cube(n,mpi_u);
    }
    else {
        MPI_Send(&i0, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&j0, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&k0, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(sub_u0, (sub_n+2)*(sub_n+2)*(sub_n+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

}

int main(int argc, char **argv)
{

    int n, i, j, k;

    /* n is size of memory allocation
     boundaries : index 0 and n-1
     inner part : index 1 to n-2
     mesh size (discretization) is 1.0/(n-1) */
    n = 16+2;

    double *u0 = malloc(sizeof(double) * n * n * n);
    double *u1 = malloc(sizeof(double) * n * n * n);
    assert(u0 != NULL && u1 != NULL);

    /* boundary is set to zero */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            u0[(i * n + j) * n + 0] = 0.0;       /* z = 0 */
            u0[(i * n + j) * n + n - 1] = 0.0;   /* z = 1 */
            u0[(i * n + 0) * n + j] = 0.0;       /* y = 0 */
            u0[(i * n + n - 1) * n + j] = 0.0;   /* y = 1 */
            u0[(0 * n + i) * n + j] = 0.0;       /* x = 0 */
            u0[((n - 1) * n + i) * n + j] = 0.0; /* x = 1 */

            u1[(i * n + j) * n + 0] = 0.0;       /* z = 0 */
            u1[(i * n + j) * n + n - 1] = 0.0;   /* z = 1 */
            u1[(i * n + 0) * n + j] = 0.0;       /* y = 0 */
            u1[(i * n + n - 1) * n + j] = 0.0;   /* y = 1 */
            u1[(0 * n + i) * n + j] = 0.0;       /* x = 0 */
            u1[((n - 1) * n + i) * n + j] = 0.0; /* x = 1 */
        }

    /* initial value for inner part is one */
    for (i = 1; i < n - 1; i++)
        for (j = 1; j < n - 1; j++)
            for (k = 1; k < n - 1; k++)
                u0[(i * n + j) * n + k] = 1.0;

    double T = 0.02;
    int nt = 200;
    double dt = T / nt;
    double dx = 1.0 / (n - 1);
    double kappa = 1.0;
    double r = kappa * dt / (dx * dx);

    if (nt < n)
    {
        fprintf(stderr,
                "nt (%d) is too small: should be >= n (%d)\n",
                nt, n);
        exit(0);
    }

    if (6.0 * r >= 1.0)
    {
        fprintf(stderr,
                "unstable condition (r=%e): nt should be larger\n",
                r);
        exit(0);
    }

    solveMPI(n, u0, r, nt, argc, argv);

    /* //plot result
    int step = (n < 30 ? 1 : n / 30); //plot about 30x30
    k = (n - 1) / 2;                  // about z = 0.5
    for (i = 0; i < n; i += step)
    {
        for (j = 0; j < n; j += step)
            printf("%e %e %e\n", i * dx, j * dx, u0[(i * n + j) * n + k]);
        printf("\n");
    } */

    /* usage: a.out > res.txt
     plot it with gnuplot: splot "res.txt" with lines */

    return 0;
}
