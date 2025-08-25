#include "stdafx.h"
#include "HMM.h"
#include <stdio.h>
#include <stdlib.h>

#define N 5
#define M 32
#define T_MAX 150

int T = 0;
long double A[N][N], B[N][M], PI[1][N], DELTA[T_MAX][N], ALPHA[T_MAX][N], BETA[T_MAX][N], XI[T_MAX][N][N], GAMMA[T_MAX][N];
int q_t_star[T_MAX], PSI[T_MAX][N];
int OB[T_MAX];

#include "printscan.c"

long double compute_forward() {
    // Initialize ALPHA
    for (int i = 0; i < N; i++) {
        ALPHA[0][i] = PI[0][i] * B[i][OB[0]];
    }
    // Induction for ALPHA
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
            ALPHA[t][j] = 0;
            for (int i = 0; i < N; i++) {
                ALPHA[t][j] += ALPHA[t - 1][i] * A[i][j] * B[j][OB[t]];
            }
        }
    }
    long double total_prob = 0;
    for (int i = 0; i < N; i++) {
        total_prob += ALPHA[T - 1][i];
    }
    return total_prob;
}

void compute_backward() {
    // Initialize BETA
    for (int i = 0; i < N; i++) BETA[T - 1][i] = 1;
    // Induction for BETA
    for (int t = T - 2; t >= 0; t--) {
        for (int i = 0; i < N; i++) {
            BETA[t][i] = 0;
            for (int j = 0; j < N; j++) {
                BETA[t][i] += A[i][j] * B[j][OB[t + 1]] * BETA[t + 1][j];
            }
        }
    }
}

long double execute_problem1() {
    long double prob = compute_forward();
    compute_backward();
    return prob;
}

long double run_viterbi() {
    // Initialize DELTA and PSI
    for (int i = 0; i < N; i++) {
        DELTA[0][i] = PI[0][i] * B[i][OB[0]];
        PSI[0][i] = -1;
    }
    // Induction for DELTA
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
            int best_idx = 0;
            long double max_prob = DELTA[t - 1][0] * A[0][j];
            for (int i = 1; i < N; i++) {
                long double current_prob = DELTA[t - 1][i] * A[i][j];
                if (current_prob > max_prob) {
                    max_prob = current_prob;
                    best_idx = i;
                }
            }
            DELTA[t][j] = max_prob * B[j][OB[t]];
            PSI[t][j] = best_idx;
        }
    }
    long double final_prob = DELTA[T - 1][0];
    int idx = 0;
    for (int i = 1; i < N; i++) {
        if (final_prob < DELTA[T - 1][i]) {
            final_prob = DELTA[T - 1][i];
            idx = i;
        }
    }
    q_t_star[T - 1] = idx;
    for (int t = T - 2; t >= 0; t--) {
        q_t_star[t] = PSI[t + 1][q_t_star[t + 1]];
    }
    return final_prob;
}

long double execute_problem2() {
    long double star_prob = run_viterbi();
    return star_prob;
}

void compute_gamma() {
    for (int t = 0; t < T; t++) {
        long double sum = 0;
        for (int i = 0; i < N; i++) {
            sum += (ALPHA[t][i] * BETA[t][i]);
        }
        for (int i = 0; i < N; i++) {
            GAMMA[t][i] = (ALPHA[t][i] * BETA[t][i]) / sum;
        }
    }
}

void compute_xi() {
    for (int t = 0; t < T - 1; t++) {
        long double sum = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                sum += (ALPHA[t][i] * A[i][j] * B[j][OB[t + 1]] * BETA[t + 1][j]);
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                XI[t][i][j] = (ALPHA[t][i] * A[i][j] * B[j][OB[t + 1]] * BETA[t + 1][j]) / sum;
            }
        }
    }
}

void reestimate_initial_prob() {
    for (int i = 0; i < N; i++) {
        PI[0][i] = GAMMA[0][i];
    }
}

void reestimate_transition_prob() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            long double xi_sum = 0;
            long double g_sum = 0;
            for (int t = 0; t < T - 1; t++) {
                xi_sum += XI[t][i][j];
                g_sum += GAMMA[t][i];
            }
            A[i][j] = xi_sum / g_sum;
        }
    }
}

void reestimate_emission_prob() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            long double numerator = 0;
            long double denominator = 0;
            for (int t = 0; t < T; t++) {
                if (OB[t] == j) {
                    numerator += GAMMA[t][i];
                }
                denominator += GAMMA[t][i];
            }
            B[i][j] = numerator / denominator;
            if (B[i][j] == 0) {
                B[i][j] = 1e-30;
            }
        }
        long double sum = 0;
        for (int j = 0; j < M; j++) {
            sum += B[i][j];
        }
        for (int j = 0; j < M; j++) {
            B[i][j] /= sum;
        }
    }
}

void execute_problem3() {
    compute_gamma();
    compute_xi();
    reestimate_initial_prob();
    reestimate_transition_prob();
    reestimate_emission_prob();
}

int _tmain(int argc, _TCHAR* argv[]) {
    const char* OBinput = "OB.txt";
    FILE* file = fopen(OBinput, "r");
    while (fscanf(file, "%d", &OB[T]) != EOF && T < 10000) {
        OB[T]--;
        T++;
    }
    fclose(file);
    const char* Ainput = "Model/A.txt";
    const char* Binput = "Model/B.txt";
    const char* PIinput = "Model/PI.txt";
    read_matrix(A, N, N, Ainput);
    READ_matrix(B, N, M, Binput);
    read_matrix(PI, 1, N, PIinput);

    int iterations = 20;

    for (int i = 1; i <= iterations; i++) {
        long double current_prob = execute_problem1();
        long double star_prob_current = execute_problem2();
        printf("Iteration: %d\n", i);
        printf("P: %e\n", current_prob);
        printf("P_STAR_PROBABILITY: %e\n", star_prob_current);
        execute_problem3();
    }

    for (int i = 0; i < T; i++) {
        printf("%d->", q_t_star[i] + 1);
    }

    const char* Aoutput = "Reestimated_Model/A.txt";
    const char* Boutput = "Reestimated_Model/B.txt";
    const char* PIoutput = "Reestimated_Model/PI.txt";
    print_matrix_to_file(A, N, N, Aoutput);
    PRINT_matrix_to_file(B, N, M, Boutput);
    print_matrix_to_file(PI, 1, N, PIoutput);
    return 0;
}