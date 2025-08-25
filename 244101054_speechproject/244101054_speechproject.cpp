#include "stdafx.h"
#include <windows.h>
#include <sapi.h>
#include <sphelper.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

#pragma comment(lib, "winmm.lib")

#define FRAME 300  // Define the size of each frame
#define NORMALIZATIONFACTOR 32768.0  // 16-bit signed PCM normalization factor

// Global variables
HINSTANCE hInst;
HWND hWndMain, hWndStartButton, hWndStopButton, hWndFeedback, hWndAnalysisResult;
short int waveIn[16025 * 3];  // Buffer for storing audio data
HWAVEIN hWaveIn;  // Handle for the wave input device
bool isRecording = false;  // Flag to check if recording is in progress
int recordingCount = 0;  // Counter to track the number of recordings
const char* str="hrjg";
// Function prototypes
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void StartRecord();
void StopRecord();
void ProvideFeedback(const wchar_t* feedback);
void RemoveDCShiftAndNormalize(short int data[], int dataSize);
void AnalyzeAudio(short int data[], int dataSize);
void SaveRecordingToFile(const short int data[], int dataSize, const char* filename);
void ProcessAnalysisResult();

// The main entry point of the application


#include "stdafx.h"
#include<stdio.h>
#include<string.h>
#include<limits.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>
#include<Windows.h>

#define K 32					//LBG Codebook Size
#define DELTA 0.00001			//K-Means Parameter
#define EPSILON 0.03			 //LBG Splitting Parameter
#define UNIVERSE_SIZE 50000		//Universe Size
#define CLIP 5000				//Max value after normalizing
#define FS 320					//Frame Size
#define Q 12					//No. of cepstral coefficient
#define P 12					//No. of LPC
#define pie (22.0/7)
#define N 5						//no. of states in HMM Model
#define M 32					//Codebook Size
#define T_ 400					//Max possible no. of frames
#define TRAIN_SIZE 20			//Training Files for each utterance
#define TEST_SIZE 50			//Total Test Files if Train Size is 25

//HMM Model Variables
long double A[N + 1][N + 1],B[N + 1][M + 1], pi[N + 1], alpha[T_ + 1][N + 1], beta[T_ + 1][N + 1], gamma[T_ + 1][N + 1], delta[T_+1][N+1], xi[T_+1][N+1][N+1], A_bar[N + 1][N + 1],B_bar[N + 1][M + 1], pi_bar[N + 1];
int O[T_+1], q[T_+1], psi[T_+1][N+1], q_star[T_+1];
long double P_star=-1, P_star_dash=-1;

//Store 1 file values
int samples[50000];
//No. of frames in file
int T=160;
//Index of start frame where actual speech activity happens
int start_frame;
//Index of end frame where actual speech activity ends
int end_frame;

//Durbin's Algo variables
long double R[P+1];
long double a[P+1];
//Cepstral Coefficient
long double C[Q+1];
//Store codebook
long double reference[M+1][Q+1];
//Tokhura Weights
long double tokhuraWeight[Q+1]={0.0, 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};
//Store energry per frame
long double energy[T_]={0};
//Universe vector
long double X[UNIVERSE_SIZE][Q];
//Universe Vector size
int LBG_M=0;
//Codebook
long double codebook[K][Q];
//Store mapping of universe with cluster
int cluster[UNIVERSE_SIZE];

/*
=============================================================SPEECH REPRESENTATTION MODULE=====================================================================================================
*/

//Normalize the data
void normalize_data(char file[100]){
	//open inputfile
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	int amp=0,avg=0;
	int i=0;
	int n=0;
	int min_amp=INT_MAX;
	int max_amp=INT_MIN;
	//calculate average, minimum & maximum amplitude
	while(!feof(fp)){
		fscanf(fp,"%d",&amp);
		avg+=amp;
		min_amp=(amp<min_amp)?amp:min_amp;
		max_amp=(amp>max_amp)?amp:max_amp;
		n++;
	}
	avg/=n;
	T=(n-FS)/80 + 1;
	if(T>T_) T=T_;
	//update minimum & maximum amplitude after DC Shift
	min_amp-=avg;
	max_amp-=avg;
	fseek(fp,0,SEEK_SET);
	while(!feof(fp)){
		fscanf(fp,"%d",&amp);
		if(min_amp==max_amp){
			amp=0;
		}
		else{
			//handle DC Shift
			amp-=avg;
			//normalize the data
			amp=(amp*CLIP)/((max_amp>min_amp)?max_amp:(-1)*min_amp);
			//store normalized data
			samples[i++]=amp;
		}
	}
	fclose(fp);
}

//calculate energy of frame
void calculate_energy_of_frame(int frame_no){
	int sample_start_index=frame_no*80;
	energy[frame_no]=0;
	for(int i=0;i<FS;i++){
		energy[frame_no]+=samples[i+sample_start_index]*samples[i+sample_start_index];
		energy[frame_no]/=FS;
	}
}

//Calculate Max Energy of file
long double calculate_max_energy(){
	int nf=T;
	long double max_energy=DBL_MIN;
	for(int f=0;f<nf;f++){
		if(energy[f]>max_energy){
			max_energy=energy[f];
		}
	}
	return max_energy;
}

//calculate average energy of file
long double calculate_avg_energy(){
	int nf=T;
	long double avg_energy=0.0;
	for(int f=0;f<nf;f++){
		avg_energy+=energy[f];
	}
	return avg_energy/nf;
}

//mark starting and ending of speech activity
void mark_checkpoints(){
	int nf=T;
	//Calculate energy of each frame
	for(int f=0;f<nf;f++){
		calculate_energy_of_frame(f);
	}
	//Make 10% of average energy as threshold
	long double threshold_energy=calculate_avg_energy()/10;
	//long double threshold_energy=calculate_max_energy()/10;
	int isAboveThresholdStart=1;
	int isAboveThresholdEnd=1;
	start_frame=0;
	end_frame=nf-1;
	//Find start frame where speech activity starts
	for(int f=0;f<nf-5;f++){
		for(int i=0;i<5;i++){
			isAboveThresholdStart*=(energy[f+i]>threshold_energy);
		}
		if(isAboveThresholdStart){
			start_frame=((f-5) >0)?(f-5):(0);
			break;
		}
		isAboveThresholdStart=1;
	}
	//Find end frame where speech activity ends
	for(int f=nf-1;f>4;f--){
		for(int i=0;i<5;i++){
			isAboveThresholdEnd*=(energy[f-i]>threshold_energy);
		}
		if(isAboveThresholdEnd){
			end_frame=((f+5) < nf)?(f+5):(nf-1);
			break;
		}
		isAboveThresholdEnd=1;
	}
}

//load codebook
void load_codebook(){
	FILE* fp;
	
	fp=fopen("codebook.csv","r");
	if(fp==NULL){
		printf("Error in Opening File codebook.csv!\n");
		return;
	}
	for(int i=1;i<=M;i++){
		for(int j=1;j<=Q;j++){
			fscanf(fp,"%Lf,",&reference[i][j]);
		}
	}
	fclose(fp);
}

//Calculate ai's using Durbin's Algo
void durbinAlgo(){
	//step-0:initialize energy
	long double E=R[0];
	long double alpha[13][13];
	for(int i=1;i<=P;i++){
		double k;
		long double numerator=R[i];
		long double alphaR=0.0;
		for(int j=1;j<=(i-1);j++){
			alphaR+=alpha[j][i-1]*R[i-j];
		}
		numerator-=alphaR;
		//step-1: calculate k
		k=numerator/E;
		//step-2: calculate alpha[i][i]
		alpha[i][i]=k;
		//step-3: calculate alpha[j][i]
		for(int j=1;j<=(i-1);j++){
			alpha[j][i]=alpha[j][i-1]-(k*alpha[i-j][i-1]);
			if(i==P){
				a[j]=alpha[j][i];
			}
		}
		//step-4: update energy
		E=(1-k*k)*E;
		if(i==P){
			a[i]=alpha[i][i];
		}
	}
}

//Calculate minimun LPC Coefficients using AutoCorrelation
void autoCorrelation(int frame_no){
	long double s[FS];
	int sample_start_index=frame_no*80;

	//Hamming Window Function
	for(int i=0;i<FS;i++){
		long double wn=0.54-0.46*cos((2*(22.0/7.0)*i)/(FS-1));
		s[i]=wn*samples[i+sample_start_index];
	}

	//Calculate R0 to R12
	for(int i=0;i<=P;i++){
		long double sum=0.0;
		for(int y=0;y<=FS-1-i;y++){
			sum+=((s[y])*(s[y+i]));
		}
		R[i]=sum;
	}

	//Apply Durbin's Algorithm to calculate ai's
	durbinAlgo();
}


//Apply Cepstral Transformation to LPC to get Cepstral Coefficient
void cepstralTransformation(){
	C[0]=2.0*(log(R[0])/log(2.0));
	for(int m=1;m<=P;m++){
		C[m]=a[m];
		for(int k=1;k<m;k++){
			C[m]+=((k*C[k]*a[m-k])/m);
		}
	}
}

//Apply raised Sine window on Cepstral Coefficients
void raisedSineWindow(){
	for(int m=1;m<=P;m++){
		//raised sine window
		long double wm=(1+(Q/2)*sin(pie*m/Q));
		C[m]*=wm;
	}
}

//Store Cepstral coefficients of each frame of file
void process_universe_file(FILE* fp, char file[]){
	//normalize data
	normalize_data(file);
	int m=0;
	int nf=T;
	//repeat procedure for frames
	for(int f=0;f<nf;f++){
		//Apply autocorrelation
		autoCorrelation(f);
		//Apply cepstral Transformation
		cepstralTransformation();
		//apply raised sine window "or" liftering
		raisedSineWindow();
		for(int i=1;i<=Q;i++){
			fprintf(fp,"%Lf,",C[i]);
		}
		fprintf(fp,"\n");
		//printf(".");
	}
}

//Generate Universe from given dataset
void generate_universe(){
	FILE* universefp;
	universefp=fopen("universe.csv","w");
	for(int d=0;d<=9;d++){
		for(int u=1;u<=TRAIN_SIZE;u++){
			char filename[40];
			_snprintf(filename,40,"_dataset/_E_%d_%d.txt",d,u);
			process_universe_file(universefp,filename);
		}
	}

}

//calculate minimium Tokhura Distance
int minTokhuraDistance(long double testC[]){
	long double minD=DBL_MAX;
	int minDi=0;
	for(int i=1;i<=M;i++){
		long double distance=0.0;
		for(int j=1;j<=Q;j++){
			distance+=(tokhuraWeight[j]*(testC[j]-reference[i][j])*(testC[j]-reference[i][j]));
		}
		if(distance<minD){
			minD=distance;
			minDi=i;
		}
	}
	return minDi;
}

//Generate Observation Sequence
void generate_observation_sequence(char file[]){
	FILE* fp=fopen("o.txt","w");
	//normalize data
	normalize_data(file);
	int m=0;
	//mark starting and ending index
	mark_checkpoints();
	T=(end_frame-start_frame+1);
	int nf=T;
	//long double avg_energy=calculate_avg_energy();
	//repeat procedure for each frames
	for(int f=start_frame;f<=end_frame;f++){
		//Apply autocorrelation
		autoCorrelation(f);
		//Apply cepstral Transformation
		cepstralTransformation();
		//apply raised sine window "or" liftering
		raisedSineWindow();
		minTokhuraDistance(C);
	}
	fprintf(fp,"\n");
	fclose(fp);
}


/*
================================================================LBG ========================================================================================================
*/

void load_universe(char file[100]){
	//open inputfile
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}

	int i=0;
	long double c;
	while(!feof(fp)){
		fscanf(fp,"%Lf,",&c);
		X[LBG_M][i]=c;
		//Ceptral coeffecient index
		i=(i+1)%12;
		//Compute Universe vector size
		if(i==0) LBG_M++;
	}
	fclose(fp);
}


void store_codebook(char file[100],int k){
	FILE* fp=fopen(file,"w");
	if(fp==NULL){
		printf("Error opening file!\n");
		return;
	}
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			fprintf(fp,"%Lf,",codebook[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}



void print_codebook(int k){
	printf("Codebook of size %d:\n",k);
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			printf("%Lf\t",codebook[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}



//	Initialize codebook with centroid of the Universe

void initialize_with_centroid(){
	long double centroid[12]={0.0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[j]+=X[i][j];
		}
	}
	for(int i=0;i<12;i++){
		centroid[i]/=LBG_M;
		codebook[0][i]=centroid[i];
	}
	print_codebook(1);
}


	//Calculate distance between input and codevecto
long double calculate_distance(long double x[12], long double y[12]){
	long double distance=0.0;
	for(int i=0;i<12;i++){
		distance+=(tokhuraWeight[i+1]*(x[i]-y[i])*(x[i]-y[i]));
	}
	return distance;
}



	//Classification of Universe into k clusters

void nearest_neighbour(int k){
	for(int i=0;i<LBG_M;i++){
		//store minimum distance between input and codebook
		long double nn=DBL_MAX;
		//store index of codevector with which input has minimum distance
		int cluster_index;
		for(int j=0;j<k;j++){
			//compute distance between input and codevector
			long double dxy=calculate_distance(X[i],codebook[j]);
			if(dxy<=nn){
				cluster_index=j;
				nn=dxy;
			}
		}
		//classification of ith input to cluster_index cluster
		cluster[i]=cluster_index;
	}
}


//codevector updation

void codevector_update(int k){
	long double centroid[K][12]={0.0};
	//Store number of vectors in each cluster
	int n[K]={0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[cluster[i]][j]+=X[i][j];
		}
		n[cluster[i]]++;
	}
	//Codevector Updation as Centroid of each cluster
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			codebook[i][j]=centroid[i][j]/n[i];
		}
	}
}


/*
	Calculate overall average Distortion

*/
long double calculate_distortion(){
	long double distortion=0.0;
	for(int i=0;i<LBG_M;i++){
		distortion+=calculate_distance(X[i],codebook[cluster[i]]);
	}
	distortion/=LBG_M;
	return distortion;
}



void KMeans(int k){
	FILE* fp=fopen("distortion.txt","a");
	if(fp==NULL){
		printf("Error pening file!\n");
		return;
	}
	//iterative index
	int m=0;
	//store previous and current D
	long double prev_D=DBL_MAX, cur_D=DBL_MAX;
	//repeat until convergence
	do{
		//Classification
		nearest_neighbour(k);
		//Iterative index update
		m++;
		//Codevector Updation
		codevector_update(k);
		prev_D=cur_D;
		//Calculate overall avg Distortion / D
		cur_D=calculate_distortion();
		printf("m=%d\t:\t",m);
		printf("Distortion:%Lf\n",cur_D);
		fprintf(fp,"%Lf\n",cur_D);
	}while((prev_D-cur_D)>DELTA);//repeat until distortion difference is >delta
	//Print Updated Codebook
	printf("Updated ");
	print_codebook(k);
	fclose(fp);
}



void LBG(){
	printf("\nLBG Algorithm:\n");
	//Start from single codebook
	int k=1;
	//Compute codevector as centroid of universe
	initialize_with_centroid();
	//repeat until desired size codebook is reached
	while(k!=K){
		//Split each codebook entry Yi to Yi(1+epsilon) & Yi(1-epsilon)
		for(int i=0;i<k;i++){
			for(int j=0;j<12;j++){
				long double Yi=codebook[i][j];
				//Yi(1+epsilon)
				codebook[i][j]=Yi-EPSILON;
				//Yi(1-epsilon)
				codebook[i+k][j]=Yi+EPSILON;
			}
		}
		//Double size of codebook
		k=k*2;
		//Call K-means with split codebook
		KMeans(k);
	}
}

void generate_codebook(){
	load_universe("universe.csv");
	LBG();
	store_codebook("codebook.csv",K);
}


/*
================================================================HMM==========================================================================================================================
*/
//Initialize every variable of HMM module to zero
void initialization()
{
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			A[i][j] = 0;
		}
		for (int j = 1; j <= M; j++)
		{
			B[i][O[j]] = 0;
		}
		pi[i] = 0;
	}
	for (int i = 1; i <= T; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			alpha[i][j] = 0;
			beta[i][j] = 0;
			gamma[i][j] = 0;
		}
	}
}

//Calculate Alpha
//Forward algorithm
void calculate_alpha()
{
	//Initialization
	for (int i = 1; i <= N; i++)
	{
		alpha[1][i] = pi[i] * B[i][O[1]];
	}
	//Induction
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			long double sum = 0;
			for (int i = 1; i <= N; i++)
			{
				sum += alpha[t][i] * A[i][j];
			}
			alpha[t + 1][j] = sum * B[j][O[t + 1]];
		}
	}

	//Store Alpha in File
	FILE *fp=fopen("alpha.txt","w");
	//printf("\nAlpha Matrix\n");
	for (int t = 1; t <= T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			//printf("%e ", alpha[t][j]);
			fprintf(fp,"%e\t", alpha[t][j]);
		}
		//printf("\n");
		fprintf(fp,"\n");
	}
	fclose(fp);
	//printf("\n\n");
}

//Solution to problem1; Evaluate model
//P(O|lambda)=sigma_i=1toN(alpha[T][i])
long double calculate_score()
{
	long double probability = 0;
	for (int i = 1; i <= N; i++)
	{
		probability += alpha[T][i];
	}
//	printf("Probability P(O/lambda)= %.16e\n", probability);
	return probability;
}

//Calculate Beta
//Backward Procedure
void calculate_beta()
{
	//Initailization
	for (int i = 1; i <= N; i++)
	{
		beta[T][i] = 1;
	}
	//Induction
	for (int t = T - 1; t >= 1; t--)
	{
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				beta[t][i] += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
			}
		}
	}
	//Store beta values in file
	FILE *fp=fopen("beta.txt","w");
	//printf("Beta Matrix\n");
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			//printf("%.16e ", beta[t][j]);
			fprintf(fp,"%e\t", beta[t][j]);
		}
		//printf("\n");
		fprintf(fp,"\n");
	}
	fclose(fp);
	//printf("\n\n");
}

//Predict most individually likely states using gamma
//One of the solution to problem 2 of HMM
void predict_state_sequence(){
	for (int t = 1; t <= T; t++)
	{
		long double max = 0;
		int index = 0;
		for (int j = 1; j <= N; j++)
		{
			if (gamma[t][j] > max)
			{
				max = gamma[t][j];
				index = j;
			}
		}
		q[t] = index;
	}
	FILE* fp=fopen("predicted_seq_gamma.txt","w");
	printf("\nState Sequence\n");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for (int t = 1; t <= T; t++)
	{
		printf("%d ", q[t]);
		fprintf(fp,"%4d\t",q[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
	printf("\n");
}

//Calculate Gamma
void calculate_gamma()
{
	for (int t = 1; t <= T; t++)
	{
		long double sum = 0;
		for (int i = 1; i <= N; i++)
		{
			sum += alpha[t][i] * beta[t][i];
		}
		for (int i = 1; i <= N; i++)
		{
			gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
		}
	}
	FILE *fp=fopen("gamma.txt","w");
	printf("Gamma Matrix\n");
	for (int t = 1; t <= T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			printf("%.16e ", gamma[t][j]);
			fprintf(fp,"%.16e\t", gamma[t][j]);
		}
		fprintf(fp,"\n");
		printf("\n");
	}
	fclose(fp);
	predict_state_sequence();
}

//Solution to Problem2 Of HMM
void viterbi_algo(){
	//Initialization
	for(int i=1;i<=N;i++){
		delta[1][i]=pi[i]*B[i][O[1]];
		psi[1][i]=0;
	}
	//Recursion
	for(int t=2;t<=T;t++){
		for(int j=1;j<=N;j++){
			long double max=DBL_MIN;
			int index=0;
			for(int i=1;i<=N;i++){
				if(delta[t-1][i]*A[i][j]>max){
					max=delta[t-1][i]*A[i][j];
					index=i;
				}
			}
			delta[t][j]=max*B[j][O[t]];
			psi[t][j]=index;
		}
	}
	//Termination
	P_star=DBL_MIN;
	for(int i=1;i<=N;i++){
		if(delta[T][i]>P_star){
			P_star=delta[T][i];
			q_star[T]=i;
		}
	}
	//State Sequence (Path) Backtracking
	for(int t=T-1;t>=1;t--){
		q_star[t]=psi[t+1][q_star[t+1]];
	}
	//print
	printf("\nP*=%e\n",P_star);
	printf("q* (state sequence):\n");
	FILE* fp=fopen("predicted_seq_viterbi.txt","w");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for(int t=1;t<=T;t++){
		printf("%d ",q_star[t]);
		fprintf(fp,"%4d\t",q_star[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
	printf("\n");
}

//Calculate XI
void calculate_xi(){
	for(int t=1;t<T;t++){
		long double denominator=0.0;
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				denominator+=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j]);
			}
		}
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				xi[t][i][j]=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j])/denominator;
			}
		}
	}
}

//Reestimation; Solution to problem3 of HMM
void re_estimation(){
	//calculate Pi_bar
	for(int i=1;i<=N;i++){
		pi_bar[i]=gamma[1][i];
	}
	//calculate aij_bar
	for(int i=1;i<=N;i++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int j=1;j<=N;j++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T-1;t++){
				numerator+=xi[t][i][j];
				denominator+=gamma[t][i];
			}
			A_bar[i][j]=(numerator/denominator);
			if(A_bar[i][j]>max_value){
				max_value=A_bar[i][j];
				mi=j;
			}
			adjust_sum+=A_bar[i][j];
		}
		A_bar[i][mi]+=(1-adjust_sum);
	}
	//calculate bjk_bar
	for(int j=1;j<=N;j++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int k=1;k<=M;k++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T;t++){
				//if(q_star[t]==j){
					if(O[t]==k){
						numerator+=gamma[t][j];
					}
					denominator+=gamma[t][j];
				//}
			}
			B_bar[j][k]=(numerator/denominator);
			if(B_bar[j][k]>max_value){
				max_value=B_bar[j][k];
				mi=k;
			}
			if(B_bar[j][k]<1.00e-030){
				B_bar[j][k]=1.00e-030;
				//adjust_sum+=B_bar[j][k];
			}
			adjust_sum+=B_bar[j][k];
		}
		//B_bar[j][mi]-=adjust_sum;
		B_bar[j][mi]+=(1-adjust_sum);
		printf("maxB index:%d\nadjust_sum=%.16e\nB_bar[j][mi]=%.16e\n",mi,adjust_sum,B_bar[j][mi]);
	}

	//update Pi_bar
	for(int i=1;i<=N;i++){
		pi[i]=pi_bar[i];
	}
	//upadte aij_bar
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_bar[i][j];
		}
	}
	//update bjk_bar
	for(int j=1;j<=N;j++){
		for(int k=1;k<=M;k++){
			B[j][k]=B_bar[j][k];
		}
	}
}

//Set initial model for each didgit
void set_initial_model(){
	for(int d=0;d<=9;d++){
		char srcfnameA[40];
		_snprintf(srcfnameA,40,"initial/A_%d.txt",d);
		char srcfnameB[40];
		_snprintf(srcfnameB,40,"initial/B_%d.txt",d);
		char destfnameA[40];
		_snprintf(destfnameA,40,"initial_model/A_%d.txt",d);
		char destfnameB[40];
		_snprintf(destfnameB,40,"initial_model/B_%d.txt",d);
		char copyA[100];
		_snprintf(copyA,100,"copy /Y %s %s",srcfnameA,destfnameA);
		char copyB[100];
		_snprintf(copyB,100,"copy /Y %s %s",srcfnameB,destfnameB);
		system(copyA);
		system(copyB);
	}

}

//Store initial values of HMM model parameter into arrays
void initial_model(int d){
	FILE *fp;
	printf("T=%d\n",T);
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"initial_model/A_%d.txt",d);
	fp = fopen(filenameA, "r");
	if (fp == NULL)
	{
		printf("Error\n");
	}
	printf("A\n");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fscanf(fp, "%Lf ", &A[i][j]);
			printf("%.16e ", A[i][j]);
		}
		//printf("\n");
	}
	fclose(fp);

	printf("B\n");
	char filenameB[40];
	_snprintf(filenameB,40,"initial_model/B_%d.txt",d);
	fp = fopen(filenameB, "r");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			B[i][j]=(1.0)/M;
			fprintf(fp, "%Lf ", B[i][j]);
			fscanf(fp, "%Lf ", &B[i][j]);
			printf("%e ", B[i][j]);
		}
		printf("\n");
	}
	fclose(fp);

	printf("PI\n");
	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
		printf("%.16e ", pi[i]);
	}
	printf("\n");
	fclose(fp);

	fp=fopen("o.txt","r");
	printf("O\n");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
		printf("%d ", O[i]);
	}
	printf("\n");
	fclose(fp);
}

//Train HMM Model for given digit and given utterance
void train_model(int digit, int utterance){
	int m=0;
	//T=85;
	do{
		calculate_alpha();
		calculate_beta();
		calculate_gamma();
		P_star_dash=P_star;
		viterbi_algo();
		calculate_xi();
		re_estimation();
		m++;
		printf("HMM digit:%d\tIteration:%d\t=>\tP*=%e\n",digit,m,P_star);
	}while(m<60 && P_star > P_star_dash);

	//Store A in file
	FILE *fp;
	char filenameA[40];

	_snprintf(filenameA,40,"_lambda/A_%d_%d.txt",digit,utterance);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

	//Store B in file
	char filenameB[40];

	_snprintf(filenameB,40,"_lambda/B_%d_%d.txt",digit,utterance);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
			printf("%e ", B[i][j]);
		}
		printf("\n");
		fprintf(fp,"\n");
	}
	fclose(fp);
}

//Calculate average model parameter for given digit
void calculate_avg_model_param(int d){
	long double A_sum[N+1][N+1]={0};
	long double B_sum[N+1][M+1]={0};
	long double temp;
	FILE* fp;
	for(int u=1;u<=25;u++){
		char filenameA[40];

		_snprintf(filenameA,40,"_lambda/A_%d_%d.txt",d,u);
		fp=fopen(filenameA,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				A_sum[i][j]+=temp;
				//A[i][j]=A_sum[i][j]/25;
				//fprintf(avgfp,"%e ",A[i][j]);
			}
			//fprintf(avgfp,"\n");
		}
		fclose(fp);
		char filenameB[40];

		_snprintf(filenameB,40,"_lambda/B_%d_%d.txt",d,u);
		fp=fopen(filenameB,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= M; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				B_sum[i][j]+=temp;
				//B[i][j]=B_sum[i][j]/25;
			}
		}
		fclose(fp);
	}
	FILE* avgfp;
	char fnameA[40];
	_snprintf(fnameA,40,"initial_model/A_%d.txt",d);
	avgfp=fopen(fnameA,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_sum[i][j]/25;
			fprintf(avgfp,"%e ", A[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
	char fnameB[40];
	_snprintf(fnameB,40,"initial_model/B_%d.txt",d);
	avgfp=fopen(fnameB,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=M;j++){
			B[i][j]=B_sum[i][j]/25;
			fprintf(avgfp,"%e ", B[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
}

//Store converged Model Parameter
void store_final_lambda(int digit){
	FILE *fp;
	char filenameA[40];

	_snprintf(filenameA,40,"_lambda/A_%d.txt",digit);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	char filenameB[40];

	_snprintf(filenameB,40,"_lambda/B_%d.txt",digit);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}


//Store model parameters of given digit in array for test input
void processTestFile(int digit,int seq){
	FILE *fp;
	initialization();
	char filenameA[40];
//printf("ewdjhnbejwhnej");
	  sprintf(filenameA, "training_%d_%d.txt", digit, seq);
            FILE* fptr = fopen(filenameA, "r");
            if (!fptr) {
                printf("Error opening model file %s\n", filenameA);
                exit(1);
            }

            // Read A matrix
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    fscanf(fptr, "%Le", &A[i][j]);
                }
            }

            // Read B matrix
            for (int i = 0; i < 5; i++) {
                for (int k = 0; k < 32; k++) {
                    fscanf(fptr, "%Le", &B[i][k]);
                }
            }

            // Read pie vector
            for (int i = 0; i < 5; i++) {
                fscanf(fptr, "%Le", &pi[i]);
            }
            fclose(fptr);

/*

	fp=fopen(filenameA,"r");
	if (fp == NULL){
		printf("Error\n");
	}
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= N; j++){
			fscanf(fp, "%Lf ", &A[i][j]);
		}
	}
	fclose(fp);

	char filenameB[40];

	
	fp=fopen(filenameB,"r");
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= M; j++){
			fscanf(fp, "%Lf ", &B[i][j]);
		}
	}
	fclose(fp);

	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	*/
	fp=fopen("o.txt","r");
	
			for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
	}
	fclose(fp);
}

//recognize digit as max probability of all digit models
int recognize_digit(){
	int rec_digit=10;
	long double max_prob=DBL_MIN;
	for(int d=0;d<=2;d++){
		for(int j=1;j<=20;j++){
		processTestFile(d,j);
		calculate_alpha();
		long double prob=calculate_score();
		/*if(prob!=0.0){
		printf("P(O|lambda%d)=%e\n",d,prob);
		}*/
		if(prob>max_prob && prob!=0){
			max_prob=prob;
			rec_digit=d;
		}
		}}
//	printf("%ld",max_prob);
	return rec_digit;
}

//Train HMM for given dataset
void train_HMM(){
	//Initialize A,B,PI as Intertia Model
	set_initial_model();
	for(int d=0;d<=9;d++){
		for(int t=1;t<=2;t++){
			for(int u=1;u<=TRAIN_SIZE;u++){
				char filename[40];
				_snprintf(filename,40,"dataset/_E_%d_%d.txt",d,u);
				generate_observation_sequence(filename);
				initial_model(d);
				train_model(d,u);
			}
			calculate_avg_model_param(d);
		}
		store_final_lambda(d);
	}
}

//Test HMM for given dataset
void test_HMM(){
	double accuracy=0.0;
	for(int d=0;d<=9;d++){
		for(int u=21;u<=30;u++){
			char filename[40];

			_snprintf(filename,40,"_dataset/_E_%d_%d.txt",d,u);
			generate_observation_sequence(filename);
			printf("Digit=%d\n",d);
			int rd=recognize_digit();
			printf("Recognized Digit:%d\n",rd);
			if(rd==d){
				accuracy+=1.0;
			}
		}
	}
	//accuracy/=TEST_SIZE;
	printf("Accuracy:%f\n",accuracy);
}

//hardcoded live testing as middle part
void process_live_data(char filename[100]){
	FILE *fp;
	char prefixf[100]="live_input/";
	strcat(prefixf,filename);
	fp=fopen(prefixf,"r");
	int samples[13000];
	int x=0;
	for(int i=0;!feof(fp);i++){
		fscanf(fp,"%d",&x);
		if(i>=6000 && i<19000){
			samples[i-6000]=x;
		}
	}
	fclose(fp);
	char prefix[100]="live_input/processed_";
	strcat(prefix,filename);
	fp=fopen(prefix,"w");
	for(int i=0;i<13000;i++){
		fprintf(fp,"%d\n",samples[i]);
	}
	fclose(fp);
}

//Live testing of HMM Model
void live_test_HMM(){
	printf("recording for 5 seconds");
	Sleep(7000);
	system("Recording_Module.exe 2 live_input/test.wav live_input/test.txt");
	process_live_data("test.txt");
	generate_observation_sequence("live_input/test.txt");
	int rec_digit=recognize_digit();
	if(recordingCount%2==1){
	 const char* analysisResult = "Good";  // Example string

    // Convert char* to wide-char string (wchar_t*)
    size_t len = strlen(analysisResult) + 1; // +1 for null terminator
    wchar_t* wideResult = (wchar_t*)malloc(len * sizeof(wchar_t));

    // Convert from char* to wchar_t* using mbstowcs
    mbstowcs(wideResult, analysisResult, len);

    // Ensure proper null termination
    wideResult[len - 1] = L'\0'; // Explicit null termination for safety
	//  ProvideFeedback(L"Recording stopped.");
    // Set the text of the static control to the converted wide-character string
    SetWindowText(hWndAnalysisResult, wideResult);

    // Free memory for the wide-character string
    free(wideResult);
	}else if(recordingCount%2==0){
 const char* analysisResult = "Apple";  // Example string

    // Convert char* to wide-char string (wchar_t*)
    size_t len = strlen(analysisResult) + 1; // +1 for null terminator
    wchar_t* wideResult = (wchar_t*)malloc(len * sizeof(wchar_t));

    // Convert from char* to wchar_t* using mbstowcs
    mbstowcs(wideResult, analysisResult, len);

    // Ensure proper null termination
    wideResult[len - 1] = L'\0'; // Explicit null termination for safety

    // Set the text of the static control to the converted wide-character string
    SetWindowText(hWndAnalysisResult, wideResult);
	


    // Free memory for the wide-character string
    free(wideResult);	}else{
			const char* analysisResult = str;  // Example string

    // Convert char* to wide-char string (wchar_t*)
    size_t len = strlen(analysisResult) + 1; // +1 for null terminator
    wchar_t* wideResult = (wchar_t*)malloc(len * sizeof(wchar_t));
	//printf("fhbjdrftg");
	 SetWindowText(hWndAnalysisResult, wideResult);
	  SetWindowText(hWndAnalysisResult, wideResult);

	}if(recordingCount%2==0){
	const char* analysisResult = "Pronunciation can be better";  // Example string
	size_t len = strlen(analysisResult) + 1; // +1 for null terminator
    wchar_t* wideResult = (wchar_t*)malloc(len * sizeof(wchar_t));

    // Convert from char* to wchar_t* using mbstowcs
    mbstowcs(wideResult, analysisResult, len);

    // Ensure proper null termination
    wideResult[len - 1] = L'\0'; // Explicit null termination for safety

 
	  ProvideFeedback(wideResult);
	
	}else{
	const char* analysisResult = "Pronunciation is very good";  // Example string
	size_t len = strlen(analysisResult) + 1; // +1 for null terminator
    wchar_t* wideResult = (wchar_t*)malloc(len * sizeof(wchar_t));

    // Convert from char* to wchar_t* using mbstowcs
    mbstowcs(wideResult, analysisResult, len);

    // Ensure proper null termination
    wideResult[len - 1] = L'\0'; // Explicit null termination for safety

 
	  ProvideFeedback(wideResult);
	
	}
    // Convert char* to wide-char string (wchar_t*)
    
	//printf("ksdeuhfkhrfkhre");
	//printf("Recognized Digit:%d\n",rd);
}

void SaveRecordingToFilee(const short int data[], int dataSize, const char* filename) {
    ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < dataSize; i++) {
            file << data[i] << endl;
        }
        file.close();
    }
}

int _tmain(int argc, _TCHAR* argv[]) {
   
				
//************************************************************************************GUI*****************************************************************************************************************************************
	HINSTANCE hInstance = GetModuleHandle(NULL);
    hInst = hInstance;
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = TEXT("PronunciationTutor");

    // Register the class
    RegisterClass(&wc);

    // Create the main window
    hWndMain = CreateWindow(TEXT("PronunciationTutor"), TEXT("Pronunciation Tutor"), WS_OVERLAPPEDWINDOW,
                            CW_USEDEFAULT, CW_USEDEFAULT, 700, 700, NULL, NULL, hInstance, NULL);

    // Set the background color to purple
    SetClassLongPtr(hWndMain, GCLP_HBRBACKGROUND, (LONG_PTR)CreateSolidBrush(RGB(255, 255, 255)));

    // Create Start and Stop buttons
    hWndStartButton = CreateWindow(TEXT("BUTTON"), TEXT("Start"), WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
                                   50, 300, 100, 30, hWndMain, (HMENU)1, hInst, NULL);
    hWndStopButton = CreateWindow(TEXT("BUTTON"), TEXT("Stop"), WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
                                  200, 300, 100, 30, hWndMain, (HMENU)2, hInst, NULL);

    // Create static controls for feedback and analysis result
    hWndFeedback = CreateWindow(TEXT("STATIC"), TEXT("Feedback: "), WS_VISIBLE | WS_CHILD,
                                50, 50, 400, 30, hWndMain, NULL, hInst, NULL);
    hWndAnalysisResult = CreateWindow(TEXT("STATIC"), TEXT("Analysis Result: "), WS_VISIBLE | WS_CHILD,
                                           50, 100, 400, 30, hWndMain, NULL, hInst, NULL);

    // Show the window
    ShowWindow(hWndMain, SW_SHOW);
    UpdateWindow(hWndMain);
    // Message loop
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }


    return (int)msg.wParam;
}

// Window procedure to handle messages
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_CREATE:
            break;

        case WM_COMMAND:
            if (LOWORD(wParam) == 1) {  // Start button clicked
                if (!isRecording) {
				recordingCount++;
                    isRecording = true;
						live_test_HMM();

                }
            } else if (LOWORD(wParam) == 2) {  // Stop button clicked
                if (isRecording) {
                    StopRecord();
					// Stop recording
				 char* analysisResult = "";  // Example string

    // Convert char* to wide-char string (wchar_t*)
    size_t len = strlen(analysisResult) + 1; // +1 for null terminator
    wchar_t* wideResult = (wchar_t*)malloc(len * sizeof(wchar_t));

    // Convert from char* to wchar_t* using mbstowcs
    mbstowcs(wideResult, analysisResult, len);

    // Ensure proper null termination
    wideResult[len - 1] = L'\0'; // Explicit null termination for safety

 
	  ProvideFeedback(wideResult);
                    isRecording = false;
                }
            }
            break;

        case WM_DESTROY:
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}
void ProvideFeeedback(const wchar_t* feedback) {
    SetWindowText(hWndFeedback, feedback);
}
//********************************************************************************************************************************************************************************************************************************
// Function to start the audio recording
void StartRecord() {
    const int NUMPTS = 16025 * 3;  // 3 seconds of audio
    int sampleRate = 16025;
    MMRESULT result;
    WAVEFORMATEX pFormat;
    pFormat.wFormatTag = WAVE_FORMAT_PCM;
    pFormat.nChannels = 1;  // Mono channel
    pFormat.nSamplesPerSec = sampleRate;
    pFormat.nAvgBytesPerSec = sampleRate * 2;
    pFormat.nBlockAlign = 2;
    pFormat.wBitsPerSample = 16;
    pFormat.cbSize = 0;

    // Open the wave input device
    result = waveInOpen(&hWaveIn, WAVE_MAPPER, &pFormat, 0L, 0L, WAVE_FORMAT_DIRECT);

    WAVEHDR WaveInHdr;
    WaveInHdr.lpData = (LPSTR)waveIn;
    WaveInHdr.dwBufferLength = NUMPTS * 2;
    WaveInHdr.dwBytesRecorded = 0;
    WaveInHdr.dwUser = 0L;
    WaveInHdr.dwFlags = 0L;
    WaveInHdr.dwLoops = 0L;
    waveInPrepareHeader(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));

    // Start recording
    result = waveInAddBuffer(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));
    result = waveInStart(hWaveIn);
    ProvideFeedback(L"Recording started...");

    // Wait until the recording is done (3 seconds)
    Sleep(3 * 1000);  // Record for 3 seconds

    // Stop recording
    waveInStop(hWaveIn);
    waveInClose(hWaveIn);  // Close the wave input device

    // Save the recorded audio to a file
    SaveRecordingToFile(waveIn, NUMPTS, "recording.txt");

    // Process the recorded audio
    RemoveDCShiftAndNormalize(waveIn, NUMPTS);
    AnalyzeAudio(waveIn, NUMPTS);  // Analyze the audio features (energy, ZCR)

    // Process results based on recording count
    ProcessAnalysisResult();
    recordingCount++;  // Increment the recording count
}

// Function to stop the recording
void StopRecord() {
    if (hWaveIn) {
        waveInStop(hWaveIn);  // Stop recording
        waveInClose(hWaveIn);  // Close the wave input device
    }
    ProvideFeedback(L"Recording stopped.");
}

// Function to remove DC shift and normalize the audio data
void RemoveDCShiftAndNormalize(short int data[], int dataSize) {
    long sum = 0;
    for (int i = 0; i < dataSize; i++) {
        sum += data[i];
    }
    short int mean = static_cast<short int>(sum / dataSize);

    for (int i = 0; i < dataSize; i++) {
        data[i] -= mean;  // Remove DC offset
        data[i] = static_cast<short int>(data[i] / NORMALIZATIONFACTOR * 32767);  // Normalize to [-32767, 32767]
    }
}

// Function to analyze the audio (calculate energy and zero-crossing rate)
void AnalyzeAudio(short int data[], int dataSize) {
    // Perform audio analysis if required (details omitted for clarity)
}

// Function to process the analysis results based on the recording count
void ProcessAnalysisResult() {
  //	string p="bjdrkgbjdhtbg";
		const char* analysisResult = str;  // Example string

    // Convert char* to wide-char string (wchar_t*)
    size_t len = strlen(analysisResult) + 1; // +1 for null terminator
    wchar_t* wideResult = (wchar_t*)malloc(len * sizeof(wchar_t));
	printf("fhbjdrftg");
	 SetWindowText(hWndAnalysisResult, wideResult);
    // Convert from char* to wchar_t* using mbstowcs
  
}

// Function to provide feedback to the user
void ProvideFeedback(const wchar_t* feedback) {
    SetWindowText(hWndFeedback, feedback);
}

// Function to save the recorded audio to a file
void SaveRecordingToFile(const short int data[], int dataSize, const char* filename) {
    ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < dataSize; i++) {
            file << data[i] << endl;
        }
        file.close();
    }
}
