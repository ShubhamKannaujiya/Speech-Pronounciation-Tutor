#pragma once

long double compute_forward();
void compute_backward();
long double execute_problem1();
long double run_viterbi();
long double execute_problem2();
void compute_gamma();
void compute_xi();
void reestimate_initial_prob();
void reestimate_transition_prob();
void reestimate_emission_prob();
void execute_problem3();