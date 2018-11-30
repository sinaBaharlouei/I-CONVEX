#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct Node{
    char base; //represents the base - A, T, C, G or '5' for start
    struct Node* parent;
    bool complete;
 //   short childId[4]; //store the index of the child in the Temp Array --> this may change after an iteration, but it's ok!
    int* childId; //store the index of the child in the Temp Array --> this may change after an iteration, but it's ok!
}node;


void BoolToChar(char *v, bool b1, bool b2);
void Gen_reads_ActLen(int *MaxLength, FILE *fpCCSIsoSeq,FILE *fpreads,FILE *fpActLength);
void Gen_dim_Len_ReadsNoisy(int MaxLength,FILE *fpreads,FILE *fpReadsNoisy,FILE *fpdimension);
void trimReads(char *InputFasta, int Reverse);
void quickSort( int a[], int l, int r);
int partition( int a[], int l, int r);
void breakReads(int np);
void getBestCenters();
void CheckFinalScore();
void CombineCentroids(int np);
void ReverseReads(FILE *fpOrig,FILE *fpReverse);
void ClusteringInfo();
void FilterAssociations();
void CopyCentroidIdRev();
void DetectCentroidLength();
int almostmedian(int x[], int n);
void AdjustCentroidLength();
double ScoreCalTotRecFast(int LOrig, int LRead, int *Orig, int *Read, double InsScore, double DelScore, double MutScore, double CorrScore);
double ScoreCalRecFast(double *RelInitScore, int *ReljIdx, int LOrig, int LRead, int *Orig, int *Read, double InsScore, double DelScore, double MutScore, double CorrScore);
void ParallelCheckFinalScoreTruncatedEnds(int my_rank, int np);



int KlargestIdx(int k, int n, double a[n], int b[k]);
double ScoreCal(int LOrig, int LRead, char Orig[LOrig], char Read[LRead], double InsScore, double DelScore, double MutScore, double CorrScore);
double ScoreCalRec(double *RelInitScore, int *ReljIdx, int LOrig, int LRead, char Orig[LOrig], char Read[LRead], double InsScore, double DelScore, double MutScore, double CorrScore, int *LEndAlignRead);
void getBreakReads(int* ActLength, int* Length, int my_rank, bool* Y, int NumReads);
void writeOutput(node** FinalCenter, int Mprime, int NumReads, int my_rank, int L, double* FinalRho, int* IdxAlgn,int *NumAlgn, double* DistAlgn);
void EMopt(int MprimeLoc, int NumReads,double* rho, int* IdxAlgn, double* DistAlgn, double* rho_loc, 
    int* BetaList, int* NumBeta, int* NumAlgn, double* TempAlpha, int n_seqs);
void NewtonEMopt(int MprimeLoc, int NumReads,double* rho, int* IdxAlgn, double* DistAlgn, double* rho_loc, int* BetaList, int* NumBeta, int* NumAlgn, double* TempAlpha, double* Hess_loc, double* Hess, double* grad_loc, double* grad );
void shift(int NumReads, double* DistAlgn, int* NumAlgn);
void calcDistAlgn(int b, int NumReads, int* ActLength, int* NumAlgn, int* StartReadIdxArray, int* Length, bool* Y, int MprimeLoc,  
    int* IdxAlgn, node** TempCenter, double* PrevScore, double* DistAlgn, int MaxNumAlgn, node** FinalCenter, int *bStartAssigning, int *MaxNumEdges, int Lcut);
void getps(int* ActLength, int r, double* InsScore, double* DelScore, double* MutScore, double*CorrScore, double* MinusEntropy);
int updateCenters(int b, int* MprimeLoc,int Mprime, double* rho, int* TopRhoIdx, node** FinalCenter, node** TempCenter, int* IndicatorTopRho, 
    int* PrevToNewIdx, int NumReads, int* NumAlgn, int* IdxAlgn, double* DistAlgn, int n_seqs, bool* completeSeqs,double threshold);
void getCompleteSeqs(int NumReads, int n_seqs, double* DistAlgn, int* IdxAlgn, int* NumAlgn, bool* completeSeqs,
    int* StartReadIdxArray, int* ActLength);
//int rearrangeSeqs(int Mprime, int MprimeLoc, int L, int b, int NumReads, bool* completeSeqs, int** FinalCenter, int* IdxAlgn, int* NumAlgn);
void Clustering(int np, int my_rank, int *bStartAssigning, int *MaxNumEdges, int Lcut);
void CalcClusterSizes();
double ScoreCalTotRec(int LOrig, int LRead, char *Orig, char *Read, double InsScore, double DelScore, double MutScore, double CorrScore, int *LMatched);
void NormalizeScores();
void Remaining();
void CombineFiles(FILE *fptarget, FILE *fp1, FILE *fp2);
void ChangeToFasta(FILE *fp, FILE *fpFasta);
void ObtainFinalCentroidId(int Rounds);
void break_matchedCenters_reads(int np);
void ParallelCheckFinalScore(int my_rank, int np);



/*constansts you may want to modify*/
//int Lcut = 35; //length of the window used to find the mapping
//int bStartAssigning[] = {20,40,60,100}; 
/*
int bStartAssigning1 = 20; 
int bStartAssigning2 = 40; //point at which you decide to discard the edges that are more unlikely
int bStartAssigning3 = 60;
int bStartAssigning4 = 100;
*/
int MaxNumAlgn = 28000; // Maximum number of aligned 
//int MaxNumEdges[] = {100, 60, 40, 20, 10};
int const sparsity = 1;
double const sparsity_power=1.03;
// int const Rounds_of_CONVEX = 1;
int const Max_Num_Clusters = 8000;
int const Reversing = 1;
int const Accuracy = 3; //1 2 3, 3 is the most accurate
double const FilteringCoverageScore = -0.4; // The threshold used to filter non-perfect edges (between node and cluster centroid)
int const INT_MAX = 1e7;

int main(int argc, char **argv)
{
	MPI_Status status; // for MPI receive function
	MPI_Init(&argc, &argv);
	srand(time(NULL));
	int size,my_rank,Lcut,MaxNumEdges[6],bStartAssigning[5],i,EndTime,startofTime,CounterRounds;
	char s[300],path[300];
	FILE *fp1,*fp2,*fp3;
	startofTime = MPI_Wtime();

	MPI_Comm_size(MPI_COMM_WORLD, &size );
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //my_rank is index of processor
	
	chdir("Clusters");
	chdir(argv[1]);
	if (my_rank == 0)
	{
		printf("Generating stop string...\n");


		char InputFasta[] = "final_cluster.fasta";



		printf("Generating read files...\n");		
		trimReads(InputFasta,Reversing);
	}

/*
	for (CounterRounds =0; CounterRounds < Rounds_of_CONVEX;CounterRounds ++)
	{
		*/
		if (my_rank ==0)
		{
			printf("Round %d: Sharing read files among cores...\n",CounterRounds);

//			char* dir = "/BreakReads";
//			char* dir1 = argv[1];

			mkdir("BreakReads",0777);
			breakReads(size);
		    mkdir("Output",0777);
		    mkdir("OutputBest",0777);
			printf("Clustering procedure starts...\n");		
		}
	    MPI_Barrier(MPI_COMM_WORLD);
	    Lcut = 35;
	    switch (Accuracy)
	    {
	    	case 1:
	    	{
				bStartAssigning[0]=8; bStartAssigning[1]=15;bStartAssigning[2] =30;bStartAssigning[3] =100;bStartAssigning[4] =200;
				MaxNumEdges[0]=50;MaxNumEdges[1]=20;MaxNumEdges[2]=10;MaxNumEdges[3]=5;MaxNumEdges[4]=5;MaxNumEdges[5]=2;
			}
	    	case 2: 
	    	{
				bStartAssigning[0]=9; bStartAssigning[1]=16;bStartAssigning[2] =50;bStartAssigning[3] =200; bStartAssigning[4] =500; 
				MaxNumEdges[0]=1000;MaxNumEdges[1]=400;MaxNumEdges[2]=50;MaxNumEdges[3]=20;MaxNumEdges[4]=10;MaxNumEdges[5]=6;
			}
	    	case 3: 
	    	{
				bStartAssigning[0]=15; bStartAssigning[1]=30;bStartAssigning[2] =60;bStartAssigning[3] =150; bStartAssigning[4] =500; 
				MaxNumEdges[0]=2500;MaxNumEdges[1]=1000;MaxNumEdges[2]=100;MaxNumEdges[3]=30;MaxNumEdges[4]=20;MaxNumEdges[5]=17;
			}
	    }

	    if (my_rank ==0)
	    {
		    printf("Before Clustering Starts\n");
	//	    getchar();
	    }
		MPI_Barrier(MPI_COMM_WORLD);
		Clustering(size,  my_rank, bStartAssigning,  MaxNumEdges, Lcut);
		MPI_Barrier(MPI_COMM_WORLD);
		
		if (my_rank == 0)
		{	
			printf("Combining centroids...\n");		
			CombineCentroids(size);
			printf("Finding best centroids...\n");		
			getBestCenters();
			printf("Calculating cluster sizes...\n");		
			CalcClusterSizes();
			printf("Breaking matched Centers and reads...\n");
			break_matchedCenters_reads(size);
			printf("Check the distance of the reads to cluster centroids...\n");		
		}
		MPI_Barrier(MPI_COMM_WORLD);	
		ParallelCheckFinalScoreTruncatedEnds(my_rank, size);

		if(my_rank == 0)
		{
			printf("Normalize the distances...\n");		
			NormalizeScores();
			printf("Filtering Associations...\n");
			FilterAssociations();
			printf("Updating CentroidId File...\n");
			CopyCentroidIdRev();
			printf("Finding best centroids...\n");		
			getBestCenters();
			printf("Calculating cluster sizes...\n");		
			CalcClusterSizes();
			printf("Compute Clustering Information...\n");
			ClusteringInfo();
			printf("Detecting and Adjusting the Length of Centroids...\n");
			DetectCentroidLength();
			AdjustCentroidLength();

			if (Reversing == 1)
			{
				printf("Reversing...\n");
				fp1 = fopen("./FinalCentersTrimmed.txt","r");
				fp2 = fopen("./FinalCentersTrimmedRev.txt","w");
				ReverseReads(fp1,fp2);
				fclose(fp1);
				fclose(fp2);				
			}	
	//		printf("Extract not properly clustered reads...\n");		
	//		mkdir("Remaining",0777);
	//		Remaining();
		}
		MPI_Barrier(MPI_COMM_WORLD);
	//	chdir("./Remaining");
//  }

	// fp1 = fopen("FinalCentroids.txt","w");
	// fclose(fp1);
	/*
 	for (CounterRounds = 0;CounterRounds<Rounds_of_CONVEX;CounterRounds++)
 	{
 		chdir("../");
 		if (my_rank ==0)
 		{
			fp1 = fopen("./FinalCentroids.txt","w");
			fp2 = fopen("./FinalCenters.txt","r");
			fp3 = fopen("./Remaining/FinalCentroids.txt","r");
			CombineFiles (fp1,fp2,fp3);
			fclose(fp1); 
			fclose(fp2);
			fclose(fp3); 		
 		}
 	}
 	
	MPI_Barrier(MPI_COMM_WORLD);
	if (my_rank == 0)
	{
		fp1 = fopen("./FinalCentroids.txt","r");
		fp2 = fopen("./Centroids.Fasta","w");
		ChangeToFasta (fp1,fp2);
		fclose(fp1);
		fclose(fp2);
		ObtainFinalCentroidId(Rounds_of_CONVEX);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	*/
    MPI_Finalize();

    return 0;

}

void FilterAssociations()
{	
   	FILE *fp1,*fp2,*fp3;
   	char vchar,s[100];
	int M,r,NumReads,tempInt;
	float tempFloat;
   	fp1 = fopen("./FinalCenters.txt", "r");
   	M = 0;
   	while (vchar != EOF)
   	{
   		if (vchar == '\n')
   			M++;
   		vchar = fgetc(fp1);
   	}
   	fclose(fp1);

   	fp1 = fopen("./dimensions.txt", "r");
	fscanf (fp1, "%d", &NumReads);
	fclose(fp1);

	fp1 = fopen("./NormalScores.txt","r");
	fp2 = fopen("./CentroidId.txt","r");
	fp3 = fopen("./CentroidIdRevised.txt","w");

	for (r=0;r<NumReads;r++)
	{
		fscanf(fp1, "%f", &tempFloat);
		fgets(s, sizeof(s), fp2);
		if ((sscanf(s,"%d",&tempInt)==1)&&(tempFloat > FilteringCoverageScore))
		{
			fprintf(fp3,"%d",tempInt);
		}
		fprintf(fp3,"\n");
	}
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);

}

void ObtainFinalCentroidId(int Rounds)
{
	FILE *fp,*fp1;
	int NumReads,i,j, M, tempCentroidId, tempRemReads;
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads);
	fclose(fp);
	char path[300] = "./";
	char s[300],vchar;
 	int *DenoisedId;  
  	DenoisedId = (int*)calloc(NumReads, sizeof(int));
 	int *DenoisedFlag;  
  	DenoisedFlag = (int*)calloc(NumReads, sizeof(int));
  	for (i=0;i<NumReads;i++)
  	{
  		DenoisedId[i] = -1;
  		DenoisedFlag[i] = 0;
  	}

   	M = 0;
	for (i=0;i<Rounds;i++)
	{
		sprintf(s,"%sRemaining/RemainingReads.txt",path);
		fp = fopen(s,"r");
		sprintf(s,"%sCentroidId.txt",path);
		fp1 = fopen(s,"r");

		for(j=0;j<NumReads;j++)
		{
			if (DenoisedFlag[j] ==0)
			{
				fscanf(fp,"%d",&tempRemReads);
				fgets(s, sizeof(s), fp1);
				sscanf(s,"%d",&tempCentroidId);
				if (tempRemReads == 0)
				{
					DenoisedFlag[j] = 1;
					DenoisedId[j] = tempCentroidId + M;
				}
			}
		}
		fclose(fp);
		fclose(fp1);

		sprintf(s,"%sFinalRho.txt",path);
	   	fp = fopen(s,"r");
	   	vchar = fgetc(fp);
	   	while (vchar != EOF)
	   	{
	   		if (vchar == '\n')
	   			M++;
	   		vchar = fgetc(fp);
	   	}
	   	fclose(fp);
		sprintf(path,"%sRemaining/",path);
		printf("Round %d of ObtainFinalCentroidId...\n",i);

	}
	fp = fopen("./ReadsCentroidID.txt","w");
	for(i=0;i<NumReads;i++)
		fprintf(fp,"%d\n",DenoisedId[i]);
	fclose(fp);

	free(DenoisedId);
	free(DenoisedFlag);
}


void CopyCentroidIdRev()
{
	FILE *fp1,*fp2;
	fp1 = fopen("CentroidIdRevised.txt","r");
	fp2 = fopen("CentroidId.txt","w");
	char v;
	v = fgetc(fp1);
	while(v != EOF)
	{
		putc(v,fp2);
		v = fgetc(fp1);
	}
	fclose(fp1);
	fclose(fp2);
	remove("./CentroidIdRevised.txt");
}


void ChangeToFasta(FILE *fp, FILE *fpFasta)
{
	int i = 100000;
	char v;
	v = fgetc(fp);
	while(v != EOF)
	{
		i++;
		fprintf(fpFasta,">Centroid%d\n",i);
		while(v != '\n')
		{
			putc(v, fpFasta);
			v = fgetc(fp);
		}
		fprintf(fpFasta,"\n");
		v = fgetc(fp);
	}
}


void CombineFiles(FILE *fptarget, FILE *fp1, FILE *fp2)
{
	char v;
	v = fgetc(fp1);
	while (v != EOF)
	{	
		putc(v, fptarget);
		v = fgetc(fp1);
	}
	v = fgetc(fp2);
	while (v != EOF)
	{	
		putc(v, fptarget);
		v = fgetc(fp2);
	}
}

void Remaining()
{
	FILE *fp1, *fp2;
	int i,np,M,L,NumReads,r,CenterIdx,tempInt;
	float temp;
	float ClusSizeThreshold = 2.5;
	float ReadScoreThreshold = -0.5;
	char vchar,s[100];
	fp1 = fopen("./dimensions.txt", "r");
	fscanf (fp1, "%d", &NumReads);
	fscanf (fp1, "%d", &L);
	fclose(fp1);


   	fp1 = fopen("FinalCenters.txt","r"); // for calculating M
   	vchar = fgetc(fp1);
   	M = 0;
   	while (vchar != EOF)
   	{
   		if (vchar == '\n')
   			M++;
   		vchar = fgetc(fp1);
   	}
   	fclose(fp1);

    int *ClusterSize;  
    ClusterSize = (int*)calloc(M, sizeof(int));
    
    char *RemainingFlag;  
    RemainingFlag = (char*)calloc(NumReads, sizeof(char));
    for(r=0;r<NumReads;r++)
    	RemainingFlag[r] = 0;
    
    
	fp1 = fopen("./ClusterSizes.txt","r");
	for (i=0;i<M;i++)
		fscanf (fp1, "%d", &ClusterSize[i]);
	fclose(fp1);
	
	// Extract the one with low scores
	fp1 = fopen("./NormalScores.txt","r");
	for (r=0;r<NumReads;r++)
	{
		fscanf (fp1, "%f", &temp);
		if(temp <ReadScoreThreshold)
			RemainingFlag[r] = 1;
	}
	fclose(fp1);
	

	fp1 = fopen("./CentroidId.txt","r");
	for(r=0;r<NumReads;r++)
	{
		fgets(s, sizeof(s), fp1);
		if (sscanf(s,"%d",&CenterIdx) == 1)
		{
			if(ClusterSize[CenterIdx] < ClusSizeThreshold)
				RemainingFlag[r] = 1;
		}
		else
		{
			RemainingFlag[r] = 1;
		}
	}
	fclose(fp1);

	
	fp1 = fopen("./ReadsNoisy.txt","r");
	fp2 = fopen("./Remaining/ReadsNoisy.txt","w");
	int RemNumReads = 0;
	for (r=0;r<NumReads;r++)
	{
		if (RemainingFlag[r] ==1)
		{
			vchar = fgetc(fp1);
			while(vchar != '\n')
			{
				putc(vchar, fp2);
				vchar = fgetc(fp1);
			}
			putc('\n',fp2);
			RemNumReads++;
		}
		else 
		{
			vchar = fgetc(fp1);
			while(vchar != '\n')
				vchar = fgetc(fp1);
		}
	}
	fclose(fp1);
	fclose(fp2);
	
	
	fp1 = fopen("./reads","r");
	fp2 = fopen("./Remaining/reads","w");
	for (r=0;r<NumReads;r++)
	{
		if (RemainingFlag[r] ==1)
		{
			vchar = fgetc(fp1);
			while(vchar != '\n')
			{
				putc(vchar, fp2);
				vchar = fgetc(fp1);
			}
			putc('\n',fp2);
		}
		else 
		{
			vchar = fgetc(fp1);
			while(vchar != '\n')
				vchar = fgetc(fp1);
		}

	}
	fclose(fp1);
	fclose(fp2);
	


	
	
	fp1 = fopen("./Remaining/dimensions.txt","w");
	fprintf (fp1, "%d\n", RemNumReads);
	fprintf (fp1, "%d\n", L);
	fclose(fp1);

	fp1 = fopen("./Remaining/RemainingSeqs.txt","w");
	for(i = 0; i < M; i++)
	{
		if (ClusterSize[i]<ClusSizeThreshold)
			fprintf(fp1,"%d\n",1);
		else 
			fprintf(fp1,"%d\n",0);
	}
	fclose(fp1);

	
	
	
	fp2 = fopen("./Remaining/RemainingReads.txt","w");
	for(r=0;r<NumReads;r++)
		fprintf(fp2,"%d\n",RemainingFlag[r]);
	fclose(fp2);


	/*
	fp1 = fopen("./Length.txt","r");	
	fp2 = fopen("./Remaining/Length.txt","w");
	for (r=0;r<NumReads;r++)
	{
		fscanf (fp1, "%d", &tempInt);
		if (RemainingFlag[r]==1)
			fprintf(fp2,"%d\n",tempInt);
	}
	fclose(fp1);
	fclose(fp2);
	*/	
	
	fp1 = fopen("./ActLength.txt","r");	
	fp2 = fopen("./Remaining/ActLength.txt","w");
	for (r=0;r<NumReads;r++)
	{
		fscanf (fp1, "%d", &tempInt);
		if (RemainingFlag[r]==1)
			fprintf(fp2,"%d\n",tempInt);
	}
	fclose(fp1);
	fclose(fp2);
	free(ClusterSize);
	free(RemainingFlag);
   
}



void NormalizeScores()
{
	FILE *fp,*fp1, *fp2;
	int NumReads,r,tempLength;
	float  temp;
	char vchar,s[100];
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads);
	fclose(fp);

	fp1 = fopen("./Scores.txt","r");
	fp2 = fopen("./ActLength.txt","r");
	fp = fopen("./NormalScores.txt","w");
	
	for(r=0;r<NumReads;r++)
	{
		fscanf (fp1, "%f", &temp);
		fscanf (fp2, "%d", &tempLength);
		fprintf(fp,"%.3f\n",temp/(tempLength + 0.1));
	}
	
	fclose(fp);
	fclose(fp1);
	fclose(fp2);
}



void CheckFinalScore()
{
	FILE *fp0, *fp, *fp1, *fp2;
	int NumReads,L,r,LRead,ell,LOrig,LMatched;
	double pins,pdel,pmut,CorrScore, MutScore,DelScore,InsScore,tempScore;
	char v;
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads);
	fscanf (fp, "%d", &L);
	fclose(fp);

  	int *ActLength;  //Actual length of reads (without added base pairs at the end)
  	ActLength = (int*)calloc(NumReads, sizeof(int));



	fp = fopen("./ActLength.txt","r");
    for(r=0;r<NumReads;r++)
    	fscanf (fp, "%d", &ActLength[r]);
    fclose(fp);
    int LengthMax = 10*L;
    

    char *Y;
    Y = (char*)calloc(LengthMax, sizeof(char));
    char *S;
    S = (char*)calloc(LengthMax, sizeof(char));

	fp = fopen("./Scores.txt", "w");
	fp0 = fopen("./EndAlign.txt", "w");
	fp1 = fopen("./reads", "r");
	fp2 = fopen("./matchedCenters.txt", "r");

    
    for (r=0;r<NumReads;r++)
    {
    	if ( r%1000 == 0)
    		printf("%d\n",r);
		if (ActLength[r]<2000)
		{
			pins = 0.018;
			pdel = 0.005;
		}
		else if(ActLength[r]<3000)
		{
			pins = 0.027;
			pdel = 0.008;
		}
		else if (ActLength[r]<5000)
		{
			pins = 0.03;
			pdel = 0.01;
		}
		else 
		{
			pins = 0.03;
			pdel = 0.02;   
		}

		//switch del and ins scores
		pmut = pdel * pins - 0.00002;
		InsScore = log(pdel);
		DelScore = log(pins);
		MutScore = log(pmut);
		CorrScore = log(1- pdel - pins - pmut);
		
	    fread((void*)(&v), sizeof(v), 1, fp1);
	    ell = 0;
	    while(v != '\n')
	    {
	    	ell++;
	        if (v=='A')
	        	S[ell] = 0;
	        else if (v == 'C')
	        	S[ell] = 1;
			else if (v == 'G')
				S[ell] = 2;
			else 
				S[ell] = 3;
			fread((void*)(&v), sizeof(v), 1, fp1);
	    }
	    LOrig = ell;
	    
	   	fread((void*)(&v), sizeof(v), 1, fp2);
	    ell = 0;
	    while(v != '\n')
	    {
	    	ell++;
	    	if (v=='A')
	    		Y[ell] = 0;
	    	else if (v == 'C')
	    		Y[ell] = 1;
	    	else if (v == 'G')
	    		Y[ell] = 2;
	    	else 
	    		Y[ell] = 3;
	    	fread((void*)(&v), sizeof(v), 1, fp2);
	    }
	    LRead = ell;
	    
	    tempScore = DelScore;
	    DelScore = InsScore;
	    InsScore = tempScore;
	    
		if (LOrig ==0)
		{
			fprintf(fp,"%.2f\n",0.0);
			fprintf(fp0,"%d\n",0);
		}
		if (LOrig > 0)
		{
			fprintf(fp,"%.2f\n", ScoreCalTotRec(LOrig,LRead, S, Y, InsScore, DelScore, MutScore, CorrScore,&LMatched));
			fprintf(fp0,"%d\n",LMatched);

		}
  	}
	fclose(fp);
	fclose(fp0);
	fclose(fp1);
	fclose(fp2);
	free(ActLength);
	free(Y);
	free(S);
}


void ParallelCheckFinalScoreTruncatedEnds(int my_rank, int np)
{
	FILE *fp0, *fp, *fp1, *fp2;
	int NumReads,L,r,LRead,ell,LOrig,temp,NumReads_tot,i,LMatched;
	int IgnoreSize = 10;
	double pins,pdel,pmut,CorrScore, MutScore,DelScore,InsScore,tempScore,temp1;
	char v,s[300];
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads_tot);
	fscanf (fp, "%d", &L);
	fclose(fp);

	temp = (int) floor((NumReads_tot+0.0)/np);
	if (my_rank < np-1) NumReads = temp;
	else NumReads = (NumReads_tot - (np-1)*temp);


  	int *ActLength;  //Actual length of reads (without added base pairs at the end)
  	ActLength = (int*)calloc(NumReads, sizeof(int));

	sprintf(s, "./BreakReads/ActLength%d.txt", my_rank);
	fp = fopen(s,"r");
    for(r=0;r<NumReads;r++)
    	fscanf (fp, "%d", &ActLength[r]);
    fclose(fp);
    int LengthMax = 10*L;
    

    char *Y;
    Y = (char*)calloc(LengthMax, sizeof(char));
    char *S;
    S = (char*)calloc(LengthMax, sizeof(char));

	sprintf(s, "./BreakReads/EndAlign%d.txt", my_rank);
	fp0 = fopen(s, "w");
	sprintf(s, "./BreakReads/Scores%d.txt", my_rank);
	fp = fopen(s, "w");
	sprintf(s, "./BreakReads/reads%d", my_rank);
	fp1 = fopen(s, "r");
	sprintf(s, "./BreakReads/matchedCenters%d.txt", my_rank);
	fp2 = fopen(s, "r");

    for (r=0;r<NumReads;r++)
    {
		if (ActLength[r]<2000)
		{
			pins = 0.018;
			pdel = 0.005;
		}
		else if(ActLength[r]<3000)
		{
			pins = 0.027;
			pdel = 0.008;
		}
		else if (ActLength[r]<5000)
		{
			pins = 0.03;
			pdel = 0.01;
		}
		else 
		{
			pins = 0.03;
			pdel = 0.02;   
		}

		//switch del and ins scores
		pmut = pdel * pins - 0.00002;
		InsScore = log(pdel);
		DelScore = log(pins);
		MutScore = log(pmut);
		CorrScore = log(1- pdel - pins - pmut);
		
	    fread((void*)(&v), sizeof(v), 1, fp1);
	    ell = 0;
	    while(v != '\n')
	    {
	    	ell++;
	        if (v=='A')
	        	S[ell] = 0;
	        else if (v == 'C')
	        	S[ell] = 1;
			else if (v == 'G')
				S[ell] = 2;
			else 
				S[ell] = 3;
			fread((void*)(&v), sizeof(v), 1, fp1);
	    }
	    LOrig = ell;
	    
	   	fread((void*)(&v), sizeof(v), 1, fp2);
	    ell = 0;
	    while(v != '\n')
	    {
	    	ell++;
	    	if (v=='A')
	    		Y[ell] = 0;
	    	else if (v == 'C')
	    		Y[ell] = 1;
	    	else if (v == 'G')
	    		Y[ell] = 2;
	    	else 
	    		Y[ell] = 3;
	    	fread((void*)(&v), sizeof(v), 1, fp2);
	    }
	    LRead = ell;
	    
	    tempScore = DelScore;
	    DelScore = InsScore;
	    InsScore = tempScore;
	    
		if (LOrig < IgnoreSize)
		{
			fprintf(fp,"%.2f\n",1000.0);
			fprintf(fp0,"%d\n",0);
		}
		if (LOrig >= IgnoreSize)
		{
        	temp1 = ScoreCalTotRec(IgnoreSize,LRead, S, Y, InsScore, DelScore, MutScore, CorrScore,&LMatched);
	    	temp1 = ScoreCalTotRec(LOrig,LRead, S, Y, InsScore, DelScore, MutScore, CorrScore,&LMatched) - temp1;
			fprintf(fp,"%.2f\n", temp1);
			fprintf(fp0,"%d\n",LMatched);
		}
  	}
  	fclose(fp0);
	fclose(fp);
	fclose(fp1);
	fclose(fp2);

	MPI_Barrier(MPI_COMM_WORLD);
	if (my_rank==0)
	{
		fp1 = fopen("Scores.txt","w");
		for(i=0;i<np;i++)
		{
			sprintf(s, "./BreakReads/Scores%d.txt", i);
			fp = fopen(s, "r");
			v = fgetc(fp);
			while(v != EOF)
			{
				putc(v,fp1);
				v = fgetc(fp);
			}
			fclose(fp);
		}
		fclose(fp1);

		fp1 = fopen("EndAlign.txt","w");
		for(i=0;i<np;i++)
		{
			sprintf(s, "./BreakReads/EndAlign%d.txt", i);
			fp = fopen(s, "r");
			v = fgetc(fp);
			while(v != EOF)
			{
				putc(v,fp1);
				v = fgetc(fp);
			}
			fclose(fp);
		}
		fclose(fp1);
	}

	free(ActLength);
	free(Y);
	free(S);
}




void ParallelCheckFinalScore(int my_rank, int np)
{
	FILE *fp0, *fp, *fp1, *fp2;
	int NumReads,L,r,LRead,ell,LOrig,temp,NumReads_tot,i,LMatched;
	double pins,pdel,pmut,CorrScore, MutScore,DelScore,InsScore,tempScore;
	char v,s[300];
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads_tot);
	fscanf (fp, "%d", &L);
	fclose(fp);

	temp = (int) floor((NumReads_tot+0.0)/np);
	if (my_rank < np-1) NumReads = temp;
	else NumReads = (NumReads_tot - (np-1)*temp);


  	int *ActLength;  //Actual length of reads (without added base pairs at the end)
  	ActLength = (int*)calloc(NumReads, sizeof(int));

	sprintf(s, "./BreakReads/ActLength%d.txt", my_rank);
	fp = fopen(s,"r");
    for(r=0;r<NumReads;r++)
    	fscanf (fp, "%d", &ActLength[r]);
    fclose(fp);
    int LengthMax = 10*L;
    

    char *Y;
    Y = (char*)calloc(LengthMax, sizeof(char));
    char *S;
    S = (char*)calloc(LengthMax, sizeof(char));

	sprintf(s, "./BreakReads/EndAlign%d.txt", my_rank);
	fp0 = fopen(s, "w");
	sprintf(s, "./BreakReads/Scores%d.txt", my_rank);
	fp = fopen(s, "w");
	sprintf(s, "./BreakReads/reads%d", my_rank);
	fp1 = fopen(s, "r");
	sprintf(s, "./BreakReads/matchedCenters%d.txt", my_rank);
	fp2 = fopen(s, "r");

    for (r=0;r<NumReads;r++)
    {
		if (ActLength[r]<2000)
		{
			pins = 0.018;
			pdel = 0.005;
		}
		else if(ActLength[r]<3000)
		{
			pins = 0.027;
			pdel = 0.008;
		}
		else if (ActLength[r]<5000)
		{
			pins = 0.03;
			pdel = 0.01;
		}
		else 
		{
			pins = 0.03;
			pdel = 0.02;   
		}

		//switch del and ins scores
		pmut = pdel * pins - 0.00002;
		InsScore = log(pdel);
		DelScore = log(pins);
		MutScore = log(pmut);
		CorrScore = log(1- pdel - pins - pmut);
		
	    fread((void*)(&v), sizeof(v), 1, fp1);
	    ell = 0;
	    while(v != '\n')
	    {
	    	ell++;
	        if (v=='A')
	        	S[ell] = 0;
	        else if (v == 'C')
	        	S[ell] = 1;
			else if (v == 'G')
				S[ell] = 2;
			else 
				S[ell] = 3;
			fread((void*)(&v), sizeof(v), 1, fp1);
	    }
	    LOrig = ell;
	    
	   	fread((void*)(&v), sizeof(v), 1, fp2);
	    ell = 0;
	    while(v != '\n')
	    {
	    	ell++;
	    	if (v=='A')
	    		Y[ell] = 0;
	    	else if (v == 'C')
	    		Y[ell] = 1;
	    	else if (v == 'G')
	    		Y[ell] = 2;
	    	else 
	    		Y[ell] = 3;
	    	fread((void*)(&v), sizeof(v), 1, fp2);
	    }
	    LRead = ell;
	    
	    tempScore = DelScore;
	    DelScore = InsScore;
	    InsScore = tempScore;
	    
		if (LOrig ==0)
		{
			fprintf(fp,"%.2f\n",0.0);
			fprintf(fp0,"%d\n",0);
		}
		if (LOrig > 0)
		{
			fprintf(fp,"%.2f\n", ScoreCalTotRec(LOrig,LRead, S, Y, InsScore, DelScore, MutScore, CorrScore,&LMatched));
			fprintf(fp0,"%d\n",LMatched);
		}
  	}
  	fclose(fp0);
	fclose(fp);
	fclose(fp1);
	fclose(fp2);

	MPI_Barrier(MPI_COMM_WORLD);
	if (my_rank==0)
	{
		fp1 = fopen("Scores.txt","w");
		for(i=0;i<np;i++)
		{
			sprintf(s, "./BreakReads/Scores%d.txt", i);
			fp = fopen(s, "r");
			v = fgetc(fp);
			while(v != EOF)
			{
				putc(v,fp1);
				v = fgetc(fp);
			}
			fclose(fp);
		}
		fclose(fp1);

		fp1 = fopen("EndAlign.txt","w");
		for(i=0;i<np;i++)
		{
			sprintf(s, "./BreakReads/EndAlign%d.txt", i);
			fp = fopen(s, "r");
			v = fgetc(fp);
			while(v != EOF)
			{
				putc(v,fp1);
				v = fgetc(fp);
			}
			fclose(fp);
		}
		fclose(fp1);
	}

	free(ActLength);
	free(Y);
	free(S);
}



double ScoreCalTotRec(int LOrig, int LRead, char *Orig, char *Read, double InsScore, double DelScore, double MutScore, double CorrScore, int *LMatched)
{
	int LcutOrig = 30;
	int jIdx = 0;
	int LcutRead = 35;
	double Score,tempdouble,RelInitScore;
	int i,ReljIdx,LReadTemp,LEndAlignRead;
	ReljIdx = -1;
	tempdouble = 0;
	for (i=0;i<LOrig - LcutOrig+1; i++)
	{
		LReadTemp = LcutRead;
		if ((LRead - ReljIdx) <LcutRead)
			LReadTemp = LRead - ReljIdx;
		jIdx = jIdx + ReljIdx + 1;
		if (jIdx<LRead)
		{
			Score = ScoreCalRec(&RelInitScore,&ReljIdx, LcutOrig,LReadTemp, &Orig[i], &Read[jIdx], InsScore, DelScore, MutScore, CorrScore, &LEndAlignRead);
			tempdouble += RelInitScore;
		}
		if (jIdx>=LRead)
		{
			jIdx = LRead+1;
			tempdouble += DelScore;
		}
	}
	if (jIdx < LRead+1)
	{
		tempdouble -= RelInitScore;
		Score += tempdouble;
		*LMatched = jIdx + LEndAlignRead;
	}
	if (jIdx == LRead+1)
	{
		Score = tempdouble +  DelScore * (LcutOrig-1);
		*LMatched = LRead;
	}
	return(Score);

}



void CalcClusterSizes()
{
	FILE *fp1, *fp2;
	int i,M,NumReads,r,temp;
	char vchar,s[100];
	fp1 = fopen("./dimensions.txt", "r");
	fscanf (fp1, "%d", &NumReads);
	fclose(fp1);

   	fp1 = fopen("FinalCenters.txt","r");
   	vchar = fgetc(fp1);
   	M = 0;
   	while (vchar != EOF)
   	{
   		if (vchar == '\n')
   			M++;
   		vchar = fgetc(fp1);
   	}
   	fclose(fp1);


	int *ClusterSize;  //Actual length of reads (without added base pairs at the end)
    ClusterSize = (int*)calloc(M, sizeof(int));
   
 

    for (i=0;i<M;i++)
    	ClusterSize[i] = 0;
	
	
	fp1 = fopen("./CentroidId.txt","r");
	fp2 = fopen("./ClusterSizes.txt","w");
	
	
	while(fscanf (fp1, "%d", &temp)!= EOF)
	{
		ClusterSize[temp]++;
	}
	
	for(i=0;i<M;i++)
	{
		fprintf(fp2,"%d\n",ClusterSize[i]);
	}
	
	fclose(fp1);
	fclose(fp2);
	free(ClusterSize);
}

void CombineCentroids(int np)
{
	FILE *fp1, *fp2;
	int i;
	char vchar,s[100];
	fp2 = fopen("./CentroidId.txt","w");

	for (i=0;i<np;i++)
	{
    	sprintf(s, "./OutputBest/CenterId%d.txt", i);
		fp1 = fopen(s,"r");
		vchar = fgetc(fp1);
		while (vchar != EOF)
		{			
		      putc(vchar, fp2);
		      vchar = fgetc(fp1);
		}
		fclose(fp1);
	}
	fclose(fp2);
}

void getBestCenters()
{

	FILE *fp,*fp1;
	int NumReads_tot,L, ell,m,r;
	char v,s[100];
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads_tot);//reads
	fscanf (fp, "%d", &L);//reads
	fclose(fp);

	int LCentroid = L-200;
  //	LCentroid = 40;

	char *S; 

    S = (char*)calloc(10000*5000, sizeof(char));
 

	fp = fopen("FinalCenters.txt","r");
	m = 0;
	v = fgetc(fp);
    while (v != EOF)
    {
    	ell = 0;
        while (v != '\n') 
        {
        	S[m*LCentroid+ell] = v;
            v = fgetc(fp); 
            ell++;
        }
        v = fgetc(fp);
        m++;
    }
    printf("Number of Centroids = %d\n",m);
    printf("NumReads_tot = %d\n",NumReads_tot);
    fclose(fp);

    fp1 = fopen("matchedCenters.txt","w");
    fp = fopen("./CentroidId.txt","r");
	for(r=0;r<NumReads_tot;r++)
	{
		fgets(s, sizeof(s), fp);
		if (sscanf(s,"%d",&m) == 1)
		{
			for(ell=0;ell<LCentroid;ell++)
			{
				putc(S[m*LCentroid+ell],fp1);
			}
		}
		putc('\n',fp1);
	}
	
	fclose(fp);
    fclose(fp1);
    free(S);


}


void Clustering(int np, int my_rank, int *bStartAssigning, int *MaxNumEdges, int Lcut)
{
	FILE *fp, *fpc;
	int M, Mprime,L,m,ell,i,k,r,j,temp,Lb,Mb,NumReads,NumReads_tot;
	int jIdx,iIdx, StartReadIdx,ReljIdx,ReadLcut,t;
	double threshold;	

	char s[100];
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads_tot);//reads
	fscanf (fp, "%d", &L);  //Length of sequences/Centers
	fclose(fp);

	threshold = 5.0/NumReads_tot;
	if (NumReads_tot<20)
           threshold = 1.0/NumReads_tot; 
	
	if (NumReads_tot<10)
           threshold = 0.5/NumReads_tot;
	
	if (NumReads_tot<6)
           threshold = 0.2/NumReads_tot;

	M = (int) 1/threshold;
	if (M>Max_Num_Clusters)
		M = Max_Num_Clusters;
	Mprime = M;
	


	Lb=5;
	char v;
    MPI_Status status; // for MPI receive function
	Mb =  (int) pow((double) 4,Lb); //Mb = 4^5, why do you need this typecasting?
	int tempCenter[Lb], Center[Mb*Lb],TopRhoIdx[Mprime];
	node **FinalCenter; //Final center is now an array of node*, each of which points to the final node of a center

    //initialize the diff processors
	int startofTime = MPI_Wtime();
	int endofTime;

	temp = (int) floor((NumReads_tot+0.0)/np);
	if (my_rank < np-1) NumReads = temp;
	else NumReads = (NumReads_tot - (np-1)*temp);

/*
    char *Y; //Y is the reads
    Y = (char*)calloc(NumReads*L, sizeof(char));
*/

    if (my_rank ==0)
    {
	    printf("Before Defining Y\n");
//	    getchar();
    }
	MPI_Barrier(MPI_COMM_WORLD);

    bool *Y;
    Y = (bool*)calloc(NumReads*2*L, sizeof(bool));

    
    int *Length;  // Length of reads
    Length = (int*)calloc(NumReads, sizeof(int));
    for (r=0;r<NumReads;r++)
    	Length[r] = L;
    
    int *ActLength;  //Actual length of reads (without added base pairs at the end)
    ActLength = (int*)calloc(NumReads, sizeof(int));
  
    getBreakReads(ActLength, Length, my_rank,Y,NumReads);
	
	srand(time(NULL)); //why do you need this random thing

	int MaxMb4Mprime = Mb; //what is this
	if (4*Mprime > Mb) MaxMb4Mprime = 4*Mprime;

	int b,mb;

   // double alpha[NumReads*Mb];

    if (my_rank ==0)
    {
	    printf("Before Defining Rho\n");
//	    getchar();
    }
	MPI_Barrier(MPI_COMM_WORLD);


    double *rho;
    rho = (double*)calloc(MaxMb4Mprime, sizeof(double));
    double *rho_loc;
    rho_loc = (double*)calloc(MaxMb4Mprime, sizeof(double));
 


    int *NumAlgn;
    NumAlgn = (int*)calloc(NumReads, sizeof(int));
    int *IdxAlgn;
    IdxAlgn = (int*)calloc(NumReads*MaxNumAlgn, sizeof(int));
    double *DistAlgn;
    DistAlgn = (double*)calloc(NumReads*MaxNumAlgn, sizeof(double));
    double *StartWindowDistAlgn;
    StartWindowDistAlgn = (double*)calloc(NumReads*100, sizeof(double));
    double *SnapshotDistAlgn;
    SnapshotDistAlgn = (double*)calloc(NumReads*100, sizeof(double));
    int *IndicatorTopRho;
    IndicatorTopRho = (int*)calloc(MaxMb4Mprime, sizeof(int));
    int *PrevToNewIdx;
    PrevToNewIdx = (int*)calloc(MaxMb4Mprime, sizeof(int));

    if (my_rank ==0)
    {
	    printf("Before Defining TempAlpha\n");
//	    getchar();
    }
	MPI_Barrier(MPI_COMM_WORLD);


    double *TempAlpha;
    TempAlpha = (double*)calloc(MaxNumAlgn, sizeof(double));

    if (my_rank ==0)
    {
	    printf("Before Defining BetaList\n");
//	    getchar();
    }
	MPI_Barrier(MPI_COMM_WORLD);

    int *BetaList;
	BetaList = (int*)calloc(MaxMb4Mprime * 40000, sizeof(int)); //max number of reads assigned to a center 

    if (my_rank ==0)
    {
	    printf("Before Defining PrevScore\n");
//	    getchar();
    }
	MPI_Barrier(MPI_COMM_WORLD);


    double *PrevScore;
	PrevScore = (double*)calloc(NumReads*MaxNumAlgn, sizeof(double));  

   if (my_rank ==0)
    {
	    printf("Before Defining StartRdIdxArr\n");
//	    getchar();
    }
	MPI_Barrier(MPI_COMM_WORLD);
	

    int *StartReadIdxArray;
    StartReadIdxArray = (int*)calloc(NumReads*MaxNumAlgn, sizeof(int));  

    if (my_rank ==0)
    {
	    printf("After StartRdIdx\n");
//	    getchar();
    }
	MPI_Barrier(MPI_COMM_WORLD);



    int *NumBeta;
    NumBeta = (int*)calloc(MaxMb4Mprime, sizeof(int)); 
    
 

    bool *completeSeqs;
    completeSeqs = (bool*)calloc(4*Mprime, sizeof(bool)); 

    FinalCenter = (node**)calloc(Mprime, sizeof(node*)); //you will have Mprime node* in the end
    
    node start;
    start.parent = NULL;
    start.base = 5;
    start.complete = false;

    /*what are these two for loops for?*/    
    for (i=0;i<Mb;i++) //Mb is 4^5?
    	rho[i] = 1.0/Mb; //estimate of frequency of each sequence -- assume uniform, divide by everything
    
    int MprimeLoc = 4; //MprimeLoc is the current number of centers (at each stage of the for loop), so we start with ACTG
    //Mprime is how many sequences/Centers you want to keep at the end
    Lb = 1;



    for(i=0;i<MprimeLoc;i++)
    {
        node* child = calloc(1,sizeof(node)); //sets all fields to null
        child->base = i; //starting centers are A,C,T,G -- finalcenter is the list of all the centers
        child->parent = &start;
        child->complete = child->parent->complete;

        child->childId = (int*) calloc (4,sizeof(int));

    	FinalCenter[i] = child; //store the pointer in Final Center

    	rho[i] = 1.0/MprimeLoc; //initial is 1/4 (assumption for simplicity)
    }

     
	double TempRho[4*Mprime]; //rho for next level --> multiply by four since there are 4 times as many centers
	node** TempCenter; //different centers that you are trying
    TempCenter = (node**)calloc(4*Mprime, sizeof(node*)); //multiply by 4 since you are trying ACTG

    //initialising stuff
    for(r = 0; r < NumReads; r++) //usef for calculating alpha
    { 
    	NumAlgn[r] = 4; //NumAlgn is the number of sequences/centers that map to the current read --> initialise with 4
    	for(j = 0; j < NumAlgn[r]; j++)
    	{
    		IdxAlgn[r + NumReads*j] = j; // set index of the sequences that align to read (initially is all the centers)
    		PrevScore[r + NumReads * j] = 0; //set the previous score to 0
    		StartReadIdxArray[r + NumReads*j] = -1; //set the previous starting index to be -1? something to do with the window
    	}
    }

	int n_incomplete = MprimeLoc; //at the start all the sequences are incomplete
    int n_seqs = MprimeLoc; //keeps track of the current number of sequences - may diverge from MprimeLoc during the for loop

    char* toReverse = (char*)malloc(L);

    double elapsedTimeInitialStuff,elapsedTimeEM, startTime,endTime, elapsedTimeCalcDist;
    double elapsedTimegetCompleteSeqs, elapsedTimeShift, elapsedTimeUpdateCenters;
    elapsedTimeShift = 0.0;
    elapsedTimeEM = 0.0;
    elapsedTimeUpdateCenters = 0.0;
    elapsedTimegetCompleteSeqs = 0.0;
    elapsedTimeCalcDist = 0.0;
    elapsedTimeInitialStuff = 0.0;



    //this for loop generates all the final centers and the calculations of rho
    int LCentroid = L-200;
    char character;
  //    LCentroid = 40;
    node* cur;
    int curCentCounter;


    for (b = 1; b < LCentroid; b++) //b is the length of the possible sequences we have generated so far
    {
		startTime =  MPI_Wtime();
    	if (my_rank == 0)// get only processor 0 to report
    	{
	if (b == LCentroid - 1) {
	    
	    printf("Length = %d out of %d\n",b,LCentroid);//rank of the processor, only get the first processor to report
		    endofTime = MPI_Wtime();
	    printf("Elapsed time = %d\n", endofTime - startofTime);
	    printf("MprimeLoc is %d, n_seqs is %d, n_incomplete is %d\n", MprimeLoc, n_seqs, n_incomplete);
	    printf("elapsedTimeEM = %.1f\n",elapsedTimeEM);
	    printf("elapsedTimeShift = %.1f\n",elapsedTimeShift);
	    printf("elapsedTimeInitialStuff = %.1f\n",elapsedTimeInitialStuff);
	    printf("elapsedTimeCalcDist = %.1f\n",elapsedTimeCalcDist);
	    printf("elapsedTimegetCompleteSeqs = %.1f\n",elapsedTimegetCompleteSeqs);
	    printf("elapsedToimeU[dateCenters= %.1f\n\n",elapsedTimeUpdateCenters);

            printf("The first center is \n");

            cur = FinalCenter[0];
            for (i = 0; i < b; i++) {
                toReverse[i] = cur->base;
                cur = cur->parent;
            }

            i--;
            for (; i >= 0; i--) {
                character = toReverse[i];
                if (character == 0)  printf("%c", 'A');
                else if (character == 1) printf("%c", 'C');
                else if (character == 2) printf("%c", 'T');
                else if (character == 3) printf("%c", 'G');
            }
            printf("\n");
        
	}
	    
    	}

        n_seqs = 4*n_incomplete + (MprimeLoc - n_incomplete); //this is the new number of temp centers you have, based on updated MprimeLoc
        curCentCounter = 0;
        //expand each of the current centers by 4
        for (mb = 0; mb < MprimeLoc; mb++){  
            for (i = 0; i < 4; i++){ 
                node* child = calloc(1,sizeof(node)); //sets all fields to null
                child->base = i; //starting centers are A,C,T,G -- finalcenter is the list of all the centers
                child->parent = FinalCenter[mb]; //set the parent to be the FinalCenter that spawned this child
                child->complete = child->parent->complete; //if the parent was complete, the child is complete too
               
                child->childId = (int*) calloc (4,sizeof(int));

                if((b>4)&&(child->parent->parent->childId != NULL))
                {
	                free(child->parent->parent->childId);
	                child->parent->parent->childId = NULL;
                }

                FinalCenter[mb]->childId[i] = curCentCounter; //store the child's index in TempCenter into the parent, for use later

                TempCenter[curCentCounter++] = child; //store the pointer in Temp Center
				
                if (child->complete) 
            	{ 
            		break; //if the child is complete, break since we only want to expand by A
                }

            }
        }
		endTime =  MPI_Wtime();
		elapsedTimeInitialStuff = elapsedTimeInitialStuff + endTime - startTime;
        //Suppose Final Center looked like this... [IICCCII] where I is an incomplete node, C is a complete node
        //TempCenter looks like this: [ACTGACTGAAAACTGACTG]

        // start calculating the scores using the window method and deciding which edges to keep between node and edge
        startTime =  MPI_Wtime();
        calcDistAlgn(b, NumReads, ActLength, NumAlgn, StartReadIdxArray, Length, 
            Y, MprimeLoc, IdxAlgn, TempCenter, PrevScore, DistAlgn, MaxNumAlgn, FinalCenter,bStartAssigning, MaxNumEdges, Lcut);
        endTime =  MPI_Wtime();
        elapsedTimeCalcDist = elapsedTimeCalcDist + endTime - startTime;

        //from the edge and MPIreduce, identifies which sequences are complete in the boolean array
        startTime =  MPI_Wtime();
        getCompleteSeqs(NumReads, n_seqs, DistAlgn, IdxAlgn, NumAlgn, completeSeqs, StartReadIdxArray, ActLength); 

        endTime =  MPI_Wtime();
        elapsedTimegetCompleteSeqs = elapsedTimegetCompleteSeqs + endTime - startTime;

        startTime =  MPI_Wtime();
        shift(NumReads, DistAlgn, NumAlgn); //Scaling and shifting DistAlgn - for computational purposes
        endTime =  MPI_Wtime();
        elapsedTimeShift = elapsedTimeShift + endTime - startTime;

  

        //calculates which centers maximise the expectation
        startTime =  MPI_Wtime();
        EMopt(MprimeLoc,NumReads,rho,IdxAlgn,DistAlgn, rho_loc, BetaList, NumBeta, NumAlgn, TempAlpha, n_seqs);
        endTime =  MPI_Wtime();
        elapsedTimeEM = elapsedTimeEM + endTime - startTime;



        //keeps the centers identified as the best by the previous steps
        startTime =  MPI_Wtime();
        n_incomplete = updateCenters(b, &MprimeLoc, Mprime, rho, TopRhoIdx, FinalCenter, TempCenter, IndicatorTopRho, 
            PrevToNewIdx, NumReads, NumAlgn, IdxAlgn, DistAlgn, n_seqs, completeSeqs, threshold); //this ones modifies MprimeLoc
        endTime =  MPI_Wtime();
        elapsedTimeUpdateCenters = elapsedTimeUpdateCenters + endTime - startTime;



        //you still need to update n_incomplete!

    }

    double *FinalRho;
    FinalRho = (double*)calloc(MprimeLoc, sizeof(double));

	for (i=0;i<MprimeLoc;i++)
		FinalRho[i] = rho[TopRhoIdx[i]]; //get the top rhos from the the list of rho

    writeOutput(FinalCenter, MprimeLoc,NumReads, my_rank, LCentroid, FinalRho,IdxAlgn,NumAlgn, DistAlgn);

    free(Y);
    free(Length);
    free(ActLength);
    free(rho);
    free(rho_loc);
    free(NumAlgn);
    free(IdxAlgn);
    free(DistAlgn);
    free(IndicatorTopRho);
    free(PrevToNewIdx);
    free(TempAlpha);
    free(BetaList);
    free(PrevScore);
    free(StartReadIdxArray);
    free(NumBeta);
    free(completeSeqs);
/*
	for(i=1;i<Mprime;i++)
	{
 	    free(FinalCenter[i]);
 	    FinalCenter[i] = NULL;
 	}
    printf("test2\n");
*/
	free(FinalCenter);
	FinalCenter = NULL;
/*
    for(i=1;i<4*Mprime;i++)
    {
	    free(TempCenter[i]);
	    TempCenter[i] = NULL;
    }
   */
	free(TempCenter);
	TempCenter = NULL;

    free(toReverse);
}

//rmb, you need to update the compelte fields as well
//make this return the number of incomplete sequences instead
int updateCenters(int b, int* MprimeLoc, int Mprime, double* rho, int* TopRhoIdx, node** FinalCenter, node** TempCenter, int* IndicatorTopRho, 
    int* PrevToNewIdx, int NumReads, int* NumAlgn, int* IdxAlgn, double* DistAlgn, int n_seqs, bool* completeSeqs, double threshold)
{

    int i,j,r,k;

    *MprimeLoc = (n_seqs < Mprime) ? n_seqs : Mprime; //set MprimeLoc = min(n_seqs, Mprime)
    bool* tempCompleteSeqs = (bool*) calloc(4*Mprime, sizeof(bool)); //you want this to be big enough


	KlargestIdx(*MprimeLoc, n_seqs, rho, TopRhoIdx); //selects the indices of the Centers with the best rho into TopRhoIdx
	//if (sparsity==1)
	//	threshold = pow(threshold,sparsity_power);


	j=0;
	for (i=0;i<*MprimeLoc; i++)
	{
		if(rho[TopRhoIdx[i]]>threshold)
			j++;
	}
	*MprimeLoc = j;


	int n_incomplete = 0;
/*
    for (i = 0; i < *MprimeLoc; i++){ //copy over up to MprimeLoc centers
        tempCompleteSeqs[i] = (*completeSeqs)[TopRhoIdx[i]];  //keep track of which sequences are complete
        FinalCenter[i] = TempCenter[TopRhoIdx[i]];
        FinalCenter[i]->complete = tempCompleteSeqs[i]; //update the complete field so we know what to expand and what not to
        if (!tempCompleteSeqs[i]) n_incomplete++; //count the nubmer of incomplete sequences
    }
    free(*completeSeqs);
    *completeSeqs = tempCompleteSeqs; 
 */
    for (i = 0; i < *MprimeLoc; i++){ //copy over up to MprimeLoc centers
        tempCompleteSeqs[i] = completeSeqs[TopRhoIdx[i]];  //keep track of which sequences are complete
        FinalCenter[i] = TempCenter[TopRhoIdx[i]];
        FinalCenter[i]->complete = tempCompleteSeqs[i]; //update the complete field so we know what to expand and what not to
        if (!tempCompleteSeqs[i]) n_incomplete++; //count the nubmer of incomplete sequences
    }

    for (i=0;i<*MprimeLoc;i++)
    	completeSeqs[i] = tempCompleteSeqs[i];
    


    //update the indices of those nodes that you want to keep 
    for (i = 0; i < n_seqs; i++) //for all the temporary centers, set indicator to 0 -- must always reset this
        IndicatorTopRho[i] = 0;

    for (i = 0; i < *MprimeLoc; i++){
        IndicatorTopRho[TopRhoIdx[i]] = 1;//set 1 for the best centers
        PrevToNewIdx[TopRhoIdx[i]] = i; //sets a new id for the selected centers
    }

    for(r = 0; r < NumReads; r++) //for each read - this code updates the edges associated with each read
    {
        k = 0;
        for (j = 0; j < NumAlgn[r]; j++) //go through each node associated with a particular read
        {
            if (IndicatorTopRho[IdxAlgn[r+NumReads*j]] == 1) //if the node is useful
            {
                IdxAlgn[r+NumReads*k] = IdxAlgn[r+NumReads*j]; //updates the edges to keep based on what Nodes there are
                DistAlgn[r+NumReads*k] = DistAlgn[r+NumReads*j]; //and the edge weights/distances
                k++;
            }
        }
        NumAlgn[r] = k; // update the number of nodes aligned to the read

        for(j = 0;j < NumAlgn[r]; j++) {
            IdxAlgn[r+NumReads*j] = PrevToNewIdx[IdxAlgn[r+NumReads*j]]; //update edges to keep to the new IDs
        }
    }

    free(tempCompleteSeqs);
    return n_incomplete;      
}

typedef struct{
    double TempDistAlgn; //its score
    int TempIdxAlgn; //index of the center linked to the read
    double TempPrevScore; //its previous score
    int TempStartReadIdx; //the start of the window
}edge;

int compareEdge(const void* p1, const void* p2){
    double result = ((edge*)p2)->TempDistAlgn - ((edge*)p1)->TempDistAlgn;
    if (result > 0.0) return 1;
    else if (result < 0.0) return -1;
    else return 0;
}

//fills out the array completeSeqs that tells you which centers are complete
void getCompleteSeqs(int NumReads, int n_seqs, double* DistAlgn, int* IdxAlgn, int* NumAlgn, bool* completeSeqs,
    int* StartReadIdxArray, int* ActLength)

{
    double score;
    double *CompletedScoreLoc;
    CompletedScoreLoc = (double*)calloc(n_seqs, sizeof(double)); 
    double *TotalScoreLoc;
    TotalScoreLoc = (double*)calloc(n_seqs, sizeof(double)); //keeps track of the total sum of scores of edges
    double *CompletedScore;
    CompletedScore = (double*)calloc(n_seqs, sizeof(double)); //keeps track of the total sum of scores of completed edges
    double *TotalScore;
    TotalScore = (double*)calloc(n_seqs, sizeof(double)); 
    int r,j;

    bool* completeEdges = (bool*)calloc(NumReads*MaxNumAlgn, sizeof(bool)); 

    //this for loop gets the CompleteEdges
    for (r = 0; r < NumReads; r++){
        for (j = 0; j < NumAlgn[r]; j++){ //if the window has gone past the length of the read, it's done
            if (StartReadIdxArray[r + NumReads*j] >= ActLength[r]) completeEdges[r + NumReads*j] = 1;
        }
    }

    for(j = 0; j < n_seqs; j++) //goes over all centers
    {
        TotalScoreLoc[j] = 0; 
        CompletedScoreLoc[j] = 0;
    }

    for (r = 0; r < NumReads; r++)
    {
        for(j = 0; j < NumAlgn[r]; j++)
        {
            score = exp(DistAlgn[r+NumReads*j]);
            TotalScoreLoc[IdxAlgn[r+NumReads*j]] += score; //calculates the scores for this connected sequence
            if (completeEdges[r+NumReads*j]) CompletedScoreLoc[IdxAlgn[r + NumReads*j]] += score; //
        }
    }

    MPI_Allreduce(CompletedScoreLoc,CompletedScore,n_seqs,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); 
    MPI_Allreduce(TotalScoreLoc,TotalScore,n_seqs,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); 

    for(j = 0; j < n_seqs; j++)
    {
        if (CompletedScore[j] > 0.9 * TotalScore[j])
        	completeSeqs[j] = true; //if more than 90% of edges (based on weight) are complete, the seq is complete
        else 
        	completeSeqs[j] = false; 
    }

    free(CompletedScoreLoc);
    free(TotalScoreLoc);
    free(CompletedScore);
    free(TotalScore);
    free(completeEdges);


}

void calcDistAlgn(int b, int NumReads, int* ActLength, int* NumAlgn,
 int* StartReadIdxArray, int* Length, bool* Y, int MprimeLoc, int* IdxAlgn, node** TempCenter, 
 double* PrevScore, double* DistAlgn, int MaxNumAlgn, node** FinalCenter, int *bStartAssigning, int *MaxNumEdges, int Lcut)
{
	char Temp1[3*Lcut],Temp2[Lcut]; //Temp1 is the window for the read, and Temp2 is the window for the sequence
    int iIdx = -1; //iIdx starts before the start of the window of the sequence
    if (b - Lcut > -1) iIdx = b-Lcut; //as b gets larger start shifting iIdx
    
    int r,j,i, ell, ReljIdx, k,StartReadIdx,ReadLcut,cur,toKeep,LEndAlignRead;
   	double tempdouble, InsScore,DelScore,MutScore,CorrScore,MinusEntropy, RelInitScore;
   	node* cur_node;
 
   	ReljIdx = 0;
	RelInitScore = 0.0; //this relative thing allows us to use the sliding window (things before the window are summarised by the relative score)

    edge *TempEdges = (edge*)calloc(MaxNumAlgn,sizeof(edge)); //a temporary array of edges for the current read
    
    for (r = 0; r < NumReads; r++) //for each read
    {       
    	k = 0; //number of edges initially 0 for each read
        getps(ActLength, r, &InsScore, &DelScore, &MutScore, &CorrScore, &MinusEntropy); //calculate the scores

    	for (j = 0; j < NumAlgn[r]; j++) //go through all the sequences/Centers possibly aligned with the read
    	{	 

            StartReadIdx = StartReadIdxArray[r + NumReads*j]; //StartReadIdx is the starting index of the window
            ReadLcut = StartReadIdx +   Lcut/3 + Lcut ; //ReadLcut is the ending index of the window
            if (Length[r] < ReadLcut) ReadLcut = Length[r];   //if the window exceeds the read length, make it fit the read

            for (ell = StartReadIdx+1; ell <=  ReadLcut; ell++)
            { //only need to copy this once, since it's the same
                BoolToChar(&Temp1[ell-StartReadIdx-1], Y[r + NumReads*(2*ell)], Y[r + NumReads*(2*ell+1)]); //Temp1 is the window of the read, which is the same
            }

            cur = (b+1) - (iIdx + 1) - 1; //we start cur at the last element of Temp2
            cur_node = TempCenter[FinalCenter[IdxAlgn[r + NumReads*j]]->childId[0]]; //start at the bottom of the trie, at the first child

            for(ell = iIdx + 1; ell < b+1; ell++){ //Temp 2 is the window of the sequence, b is the length of the sequences generated so far
                Temp2[cur--] = cur_node->base; //filling it from back to front, so Temp2 is now in the right order
                cur_node = cur_node->parent; 
            }

            for (i = 0; i < 4; i++)
            { //performs the score scalculations
    				      			
                if (FinalCenter[IdxAlgn[r + NumReads*j]]->complete && i > 0 ) break; // if it's complete, only examine once

                Temp2[(b+1) - (iIdx + 1) - 1] = (char) i; //only the last base of Temp2 actually changes

                //b is the length of the sequence we have copied so far - if the length is less than the window, just calculate the score
    			if (b < Lcut-1) tempdouble =  ScoreCal(b-iIdx, ReadLcut-StartReadIdx, Temp2, Temp1, InsScore,  DelScore,  MutScore,  CorrScore);
    		
                //if b is bigger than the window, need to shift the window and use the previous scores that you had
    			if (b >= Lcut - 1) 
    			{ //faster because it's not calculating over the full length, only the length of the window (b-iIdx, ReadLcut etc.)
        			tempdouble =  ScoreCalRec(&RelInitScore, &ReljIdx, b-iIdx, ReadLcut-StartReadIdx, Temp2, Temp1,  InsScore,  DelScore,  MutScore,  CorrScore,&LEndAlignRead); 
    				tempdouble += PrevScore[r + NumReads*j]; //add the previous score to the current score of the window
    			}

                //store the index of this new center, which is stored in its parent!
                TempEdges[k].TempIdxAlgn = FinalCenter[IdxAlgn[r + NumReads*j]]->childId[i]; 
    			TempEdges[k].TempDistAlgn = tempdouble;

    			if (b >= Lcut - 1) //if b is bigger than the window
    			{
                    TempEdges[k].TempPrevScore = PrevScore[r + NumReads*j] + RelInitScore;	//update prevscore for the edge
                    TempEdges[k].TempStartReadIdx = StartReadIdxArray[r + NumReads*j] + ReljIdx + 1; //advance the window for the edge to align -- always add 1 since you want the window to advance for the next iteration
    			}

    			k++; //k is how many edges to the read you kept in total	
            }

    	}

        if (b < bStartAssigning[0]) toKeep = MaxNumEdges[0];
        else if (b < bStartAssigning[1]) toKeep = MaxNumEdges[1];
        else if (b < bStartAssigning[2]) toKeep = MaxNumEdges[2];
        else if (b < bStartAssigning[3]) toKeep = MaxNumEdges[3];
        else if (b < bStartAssigning[4]) toKeep = MaxNumEdges[4];
        else toKeep = MaxNumEdges[5];

    	NumAlgn[r] = toKeep < k ? toKeep : k; //sets the number of edges to be min(k, MaxNumEdges)
        qsort(TempEdges, k, sizeof(edge), compareEdge); //sort the sequences by descending order of score
        //choose the best NumAlgn of them 

    	for(j = 0;j < NumAlgn[r]; j++) //keep the top k/MaxNumedges edges for consideration
    	{
    		DistAlgn[r + NumReads*j] = TempEdges[j].TempDistAlgn; //store the distance for the current read and the jth sequence aligned to it
    		IdxAlgn[r + NumReads*j] = TempEdges[j].TempIdxAlgn; //store the index of the jth sequence aligned to the current read
    		StartReadIdxArray[r + NumReads*j] = -1;
    		PrevScore[r + NumReads*j] = 0; //when b is less than Lcut, the previous score is 0
    		if (b >= Lcut-1)
    		{ 
				StartReadIdxArray[r + NumReads*j] = TempEdges[j].TempStartReadIdx; 
				PrevScore[r+NumReads*j] = TempEdges[j].TempPrevScore;
    		}
    	}  


    } //end of r for loop /calculating for all the reads

   free(TempEdges);
   cur_node = NULL;
   free(cur_node);
   
}

//calculates the probabilistic weights based on some kind of prior knowledge
void getps(int* ActLength, int r, double* InsScore, double* DelScore, double* MutScore, double*CorrScore, double* MinusEntropy){
    double pdel,pins,pmut;

        if (ActLength[r]<2000)
        {
            pins = 0.018;
            pdel = 0.005;
        }
        else if(ActLength[r]<3000)
        {
            pins = 0.027;
            pdel = 0.008;
        }
        else if (ActLength[r]<5000)
        {
            pins = 0.03;
            pdel = 0.01;
        }
        else 
        {
            pins = 0.03;
            pdel = 0.02;   
        }
        // Remove this for future
 		//       pdel = 0.01;
  		//      pins = 0.01;

        pmut = pdel * pins - 0.00002;
        *InsScore = log(pdel);
        *DelScore = log(pins);
        *MutScore = log(pmut);
        *CorrScore = log(1- pdel - pins - pmut);
        *MinusEntropy =  pdel * log(pdel) + pins * log(pins) + pmut * log(pmut) + (1- pdel - pins - pmut) * log(1- pdel - pins - pmut);         
}


void shift(int NumReads, double* DistAlgn, int* NumAlgn){
	int i,r;
	double tempMax;
	for(r = 0; r < NumReads; r++) //for each read
	{
		tempMax = DistAlgn[r];
		for(i = 0;i < NumAlgn[r]; i++)
			if (DistAlgn[r+NumReads*i] > tempMax)
				tempMax = DistAlgn[r+NumReads*i];
		for (i = 0; i < NumAlgn[r]; i++)
			DistAlgn[r+NumReads*i] = DistAlgn[r+NumReads*i] - tempMax + 30; //add some kind of constant
	}
}



int ChangeBase(int length, int n, int *n4)
{
	int j;
	for(j = length-1; j>=0;j--)
	{
		n4[j] = n%4;
		n = n/4;
	}	
	return(0);
}



typedef struct{
	int index;
	double value;
}cell;

int compareCell(const void* p1, const void* p2){
	double result = ((cell*)p2)->value - ((cell*)p1)->value;
	if (result > 0.0) return 1;
	else if (result < 0.0) return -1;
	else return 0;
}

/*
Selects the k largest values from an array a of length n, and stores their original indices in array b in descending order
*/
int KlargestIdx(int k, int n, double *a, int *b) 
{
	int i;
	cell cells[n];
	for (i = 0; i < n; i++){ //initialise the structs
		cells[i].index = i;
		cells[i].value = a[i];
	}

	qsort(cells, n, sizeof(cell), compareCell); //sort them

	for (i = 0; i < k; i++){
		b[i] = cells[i].index;
	}

    return 0;
}

//uses DP to calculate te score --> modify to read the QV scores
double ScoreCal(int LOrig, int LRead, char *Orig, char *Read, double InsScore, double DelScore, double MutScore, double CorrScore)
{
    double Score[LOrig+1][LRead+1]; //add one row and column full of zero
    int i,j;
    double temp;
    for (i=0;i<=LOrig;i++)
        Score[i][0] = i * DelScore; //the number of bases you have to delete for a match is i
    for (j=0;j<=LRead;j++)
        Score[0][j] = j * InsScore; //the number of bases that have to be inserted into the read for there to be a match with orig( an empty string)
    
    for (i=1; i<=LOrig;i++){ //do DP
        for (j=1;j<=LRead;j++)
        {
            temp = Score[i][j-1] + InsScore; //a base was inserted into the original sequence
            if ((Score[i-1][j] + DelScore) > temp) //a base was deleted from the original sequence (in row i)
                temp = Score[i-1][j] + DelScore;
            if (Orig[i-1] == Read[j-1]) //if it's correct -- how does this work?
            {
                if ((Score[i-1][j-1] + CorrScore) > temp)
                    temp = (Score[i-1][j-1] + CorrScore); 
            }
            else
            {
                if ((Score[i-1][j-1] + MutScore) > temp)
                    temp = (Score[i-1][j-1] + MutScore); //calculate for mutation
            }       
            Score[i][j] = temp;         //always set to be the highest score we can find
        }
    }

    for (j=0;j<=LRead;j++) //find the best score in the last row
    {
        if(Score[LOrig][j] > temp)
            temp = Score[LOrig][j];
    }
// no free    free(Score);
    return(temp);
}

/*calculates the score along with the offset score for use with the window
should modify to include QValues
*/
double ScoreCalRec(double *RelInitScore, int *ReljIdx, int LOrig, int LRead, char *Orig, char *Read, double InsScore, double DelScore, double MutScore, double CorrScore, int *LEndAlignRead)
{
	double Score[LOrig+1][LRead+1]; //add one row and column full of zero
	// Ins 0; Del 1; Corr 2; Mut 3; --> this corresponds to the path taken in dynamic path
	int DynamicPath[LOrig+1][LRead+1]; //keeps track of the path to follow in the table (i.e. was it an insertion, deletion)
	int i,ii,jj,j,k,jmax,tempint,TempAligning[3*LOrig];
	double temp;
	
	
	for (i=0;i<=LOrig;i++)
	{
		Score[i][0] = i * DelScore;
		DynamicPath[i][0] = 1;
	}
	for (j=0;j<=LRead;j++)
	{
		Score[0][j] = j * InsScore;
		DynamicPath[0][j] = 0;
	}
	
	for (i = 1; i <= LOrig; i++){
		for (j = 1; j <= LRead; j++)
		{
			temp = Score[i][j-1] + InsScore; //calculate insertion
			DynamicPath[i][j] = 0; //used for tracing the dynamic path
			if ((Score[i-1][j] + DelScore) > temp)
			{
				temp = Score[i-1][j] + DelScore;
				DynamicPath[i][j] = 1;
			}
			if (Orig[i-1] == Read[j-1])
			{
				if ((Score[i-1][j-1] + CorrScore) > temp)
				{
					temp = (Score[i-1][j-1] + CorrScore);
					DynamicPath[i][j] = 2;
				}
			}
			else
			{
				if ((Score[i-1][j-1] + MutScore) > temp)
				{
					temp = (Score[i-1][j-1] + MutScore);
					DynamicPath[i][j] = 3;
				}
			}		
			Score[i][j] = temp;
		}
    }

    //up to here is just doing the same DP in ScoreCal

	jmax = LRead; // LRead is the length of the window of the read
	for (j = 0; j <= LRead; j++) //find the best score and the corresponding pos in the read, jmax
	{
		if(Score[LOrig][j] > temp)
		{
			temp = Score[LOrig][j];
			jmax = j;
		}
	}

	k = 0;
	i = LOrig;
	j = jmax;
	*LEndAlignRead = jmax;

	while((i>0)||(j>0)) //his traces the dynamic path from front to back, but what is TempAligining?
	{
		TempAligning[k] = DynamicPath[i][j]; 
		if (DynamicPath[i][j] == 0) //0 is insertion
		{
			jj = j-1;
			ii = i;
		}
		if (DynamicPath[i][j] == 1)
		{
			ii = i-1;
			jj = j;
		}
		if (DynamicPath[i][j] == 2)
		{
			jj = j-1;
			ii = i-1;
		}
		if (DynamicPath[i][j] == 3)
		{
			ii = i-1;
			jj = j-1;
		}	
		j = jj;
		i = ii;  //following the instructions above, go to the next step of the path in the grid
		k +=1;
	}


	for (i=0;i<(k-1)/2;i++) //reverses TempAligning (I think), so you get the path in the correct order
	{	
		tempint = TempAligning[k-i-1];
		TempAligning[k - i-1] = TempAligning[i];
		TempAligning[i] = tempint;
	} 

	i = 0;
	while((TempAligning[i]==0) && (i<k)) //while insertion and we haven't reached the full length?
	{
		i += 1; //increment until no longer inserting --> i is the number of insertions
	}

	*ReljIdx = i; //the offset index is how many insertions there were
	*RelInitScore = i * InsScore; //i is the number of insertions

	if (TempAligning[i] == 1) //deletion
	{
		*ReljIdx = *ReljIdx -1 ; //go back 1 since a base was deleted
		*RelInitScore = *RelInitScore + DelScore;
	}
	if (TempAligning[i] == 2) //correction
	{
		*RelInitScore = *RelInitScore + CorrScore;
	}
	if (TempAligning[i] == 3) //mutation
	{
		*RelInitScore = *RelInitScore + MutScore;
	}
	
 	// no 	free(TempAligning);
	//free(Score);
	//	free(DynamicPath);
	
	return(temp);
}

/*reads in the sequences and lengths etc. into the arrays*/
void getBreakReads(int* ActLength, int* Length, int my_rank, bool* Y, int NumReads){
	char s[100];
	char v;
	int r,ell;
	FILE *fpc;
	sprintf(s, "./BreakReads/ActLength%d.txt", my_rank);
	fpc = fopen(s,"r");
    for(r=0;r<NumReads;r++) //gets actual length
    	fscanf (fpc, "%d", &ActLength[r]);
    fclose(fpc);
	
	sprintf(s, "./BreakReads/ReadsNoisy%d.txt", my_rank);
	FILE *fp = fopen(s,"r");
	if (!fp)
	{
		printf("Unable to open file!");
	}
	if (fp)
	{
	    for ( r=0;r<NumReads;r++) //for each read
	    {
	        for ( ell=0;ell<Length[r];ell++) //for each character in the read
	        {
                fread((void*)(&v), sizeof(v), 1, fp); //get the character
                if (v=='A')
                {
            		Y[r+NumReads*2*ell] = 0;   //think of Y as a LengthMax rows by NumReads columns matrix
            		Y[r+NumReads*(2*ell+1)] = 0;
                }
                else if (v == 'C')
                {
                	Y[r+NumReads*2*ell] = 0;
                	Y[r+NumReads*(2*ell+1)] = 1;
                }
				else if (v == 'G')
                {
                	Y[r+NumReads*2*ell] = 1;
                	Y[r+NumReads*(2*ell+1)] = 0;
                }
				else 
                {
                	Y[r+NumReads*2*ell] = 1;
                	Y[r+NumReads*(2*ell+1)] = 1;
                }	        
            }
	        fread((void*)(&v), sizeof(v), 1, fp); //what's this extra read for? is it the newline?
	    }
	}
    fclose(fp);
}

void writeOutput(node** FinalCenter, int Mprime, int NumReads, int my_rank, int L, double* FinalRho, int* IdxAlgn,int *NumAlgn, double* DistAlgn){
    FILE *fp;
    int r,i,j,m;
	char s[100];
	double cur_score, best_score;
	int best_index, first;

    if(my_rank ==0)
    {   
        char* toReverse = (char*)malloc(L);

		fp = fopen("FinalCenters.txt", "w");
		for (i = 0; i < Mprime; i++)
		{

            node* cur = FinalCenter[i];
			for (j = 0; j < L; j++)
			{    
                toReverse[j] = cur->base;
                cur = cur->parent;
			}

            j--;

            for (; j >= 0; j--){
                char base = toReverse[j];

                if (base == 0)
                    fprintf(fp,"%c",'A');
                else if (base == 1)
                    fprintf(fp,"%c",'C');
                else if (base == 2)
                    fprintf(fp,"%c",'G');
                else
                    fprintf(fp,"%c",'T');
            }


			fprintf(fp,"\n");
		}
		fclose(fp);
		free(toReverse);

		fp = fopen("FinalRho.txt", "w");
		for (m = 0; m < Mprime; m++)
		{
			fprintf(fp,"%.20f\n",FinalRho[m]);
		}
		fclose(fp);
    }
    sprintf(s, "./Output/CenterId%d.txt", my_rank); //prints all
	fp = fopen(s,"w");
    for(r=0;r<NumReads;r++)
    {
		for(j=0;j<NumAlgn[r];j++)
		{
			fprintf(fp,"%d ",IdxAlgn[r+NumReads*j]);
		}
		fprintf(fp,"\n");
    }
    fclose(fp);

	sprintf(s, "./OutputBest/CenterId%d.txt", my_rank); //prints the best
	fp = fopen(s,"w");
    for(r=0;r<NumReads;r++)
    {
    	best_index = 0;
    	first = 1;

		for(j=0;j<NumAlgn[r];j++)
		{
			cur_score = DistAlgn[r + NumReads*j];
			if (first || cur_score > best_score){ //first find the index (j) of the one with the best distance
				best_index = j;
				best_score = cur_score;
				first = 0;
			}
			
		}
		if (!first) fprintf(fp,"%d ",IdxAlgn[r + NumReads*best_index]); //this prints all the reads - need to change to print only the best read
		fprintf(fp,"\n");
    }
    fclose(fp);

}


void EMopt(int MprimeLoc, int NumReads,double* rho, int* IdxAlgn, double* DistAlgn, double* rho_loc, 
    int* BetaList, int* NumBeta, int* NumAlgn, double* TempAlpha, int n_seqs)
{
	int i,MaxIter,m,t,r;
	double Sumalpha, SumRho;
	MaxIter = 6;

	for(i = 0; i < n_seqs; i++) //for each of the potential centers
		rho[i] = 1.0/n_seqs; //we start with a uniform abundance
	
	//Iterate the EM for MaxIter times - the likelihood is nondecreasing per iteration
	for(t = 0; t < MaxIter; t++)
	{
		for (m = 0; m < n_seqs; m++) //for each of the sequences 
			NumBeta[m] = 0; //NumBeta tells you how many reads each sequence maps to

		for(r = 0; r < NumReads; r++) //for each read
		{
			Sumalpha = 0;
			
            for (i = 0; i < NumAlgn[r]; i++) //r is the index of the reads, NumAlgn is the number of sequences that are aligned to the reads
			{
				//DistAlgn is the log of alpha - probability of observing read for given sequence --based on score
				TempAlpha[i] = exp(DistAlgn[r+NumReads*i]) * rho[IdxAlgn[r + NumReads*i]]; //exp(edge_score * rho_of_center)
				Sumalpha += TempAlpha[i]; //get the total sum
			}

			for(i = 0; i < NumAlgn[r]; i++)
			{
                //i think BetaList is the normalised _____ (abundance?) for each center and associated edge
//				BetaList[IdxAlgn[r + NumReads*i] + n_seqs*NumBeta[IdxAlgn[r + NumReads*i]]] = TempAlpha[i]/ Sumalpha; //normalise the numbers
				BetaList[IdxAlgn[r + NumReads*i] + n_seqs*NumBeta[IdxAlgn[r + NumReads*i]]] = floor(TempAlpha[i]/ Sumalpha * INT_MAX); //Make things integer
				//BetaList[IdxAlgn,NumBeta[IdxAlgn]]
				NumBeta[IdxAlgn[r + NumReads*i]] += 1; 
			}

		}
		
		for (m = 0; m < n_seqs; m++)
		{
			rho_loc[m] = 0; //get the sum of things for the rho local to the processor
			for (i = 0; i < NumBeta[m]; i++) //for each edge the center is connected to
				rho_loc[m] += (BetaList[m + n_seqs*i]+0.0)/INT_MAX; //BetaList[m,i] --> get the total abundance for that particular center and edge
		}

		//MPI_Allreduce(rho_loc,rho,n_seqs,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); //combines the rho across different processors to give
		//global value of rho

		for(m = 0; m < n_seqs; m++)
			rho[m] = rho_loc[m];
        //this step is the same as dividing each abundance by R (NumReads), since all edges from each read contributes 1
		SumRho = 0;
		for(m = 0; m < n_seqs; m++)
			SumRho += rho[m];
		for (m = 0; m < n_seqs; m++) //normalise the numbers once again --> 
			rho[m] = rho[m]/SumRho;       	
	}
	MPI_Allreduce(rho_loc,rho,n_seqs,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	SumRho = 0;
	for(m = 0; m < n_seqs; m++)
		SumRho += rho[m];
	for (m = 0; m < n_seqs; m++) //normalise the numbers once again --> 
		rho[m] = rho[m]/SumRho;  


	MaxIter = 9;
   // Start EM
	for(t = 0; t < MaxIter; t++)
	{
		for (m = 0; m < n_seqs; m++) //for each of the sequences 
			NumBeta[m] = 0; //NumBeta tells you how many reads each sequence maps to

		for(r = 0; r < NumReads; r++) //for each read
		{
			Sumalpha = 0;
			
            for (i = 0; i < NumAlgn[r]; i++) //r is the index of the reads, NumAlgn is the number of sequences that are aligned to the reads
			{
				//DistAlgn is the log of alpha - probability of observing read for given sequence --based on score
				TempAlpha[i] = exp(DistAlgn[r+NumReads*i]) * rho[IdxAlgn[r + NumReads*i]]; //exp(edge_score * rho_of_center)
				Sumalpha += TempAlpha[i]; //get the total sum
			}

			for(i = 0; i < NumAlgn[r]; i++)
			{
                //i think BetaList is the normalised _____ (abundance?) for each center and associated edge
//				BetaList[IdxAlgn[r + NumReads*i] + n_seqs*NumBeta[IdxAlgn[r + NumReads*i]]] = TempAlpha[i]/ Sumalpha; //normalise the numbers
				BetaList[IdxAlgn[r + NumReads*i] + n_seqs*NumBeta[IdxAlgn[r + NumReads*i]]] = floor(TempAlpha[i]/ Sumalpha * INT_MAX); //Make things integer
				//BetaList[IdxAlgn,NumBeta[IdxAlgn]]
				NumBeta[IdxAlgn[r + NumReads*i]] += 1; 
			}

		}

		for (m = 0; m < n_seqs; m++)
		{
			rho_loc[m] = 0; //get the sum of things for the rho local to the processor
			for (i = 0; i < NumBeta[m]; i++) //for each edge the center is connected to
				rho_loc[m] += (BetaList[m + n_seqs*i]+0.0)/INT_MAX; //BetaList[m,i] --> get the total abundance for that particular center and edge
		}

		MPI_Allreduce(rho_loc,rho,n_seqs,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); //combines the rho across different processors to give
		//global value of rho

        //this step is the same as dividing each abundance by R (NumReads), since all edges from each read contributes 1
		SumRho = 0;
		for(m = 0; m < n_seqs; m++)
			SumRho += rho[m];
		for (m = 0; m < n_seqs; m++) //normalise the numbers once again --> 
			rho[m] = rho[m]/SumRho; 

		if (sparsity==1)
			for(m=0; m<n_seqs;m++)
				rho[m] = pow(rho[m],sparsity_power);
	
	}
 
}


void break_matchedCenters_reads(int np)
{
	FILE *fp, *fp1;
	int NumReads,n,i,blcklngth;
	char vchar;
	char s[100];
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads);
	fclose(fp);

    blcklngth = (int) floor((NumReads+0.0)/np);
    fp = fopen("./matchedCenters.txt","r");
    for (n=0;n<np-1;n++)
    {
    	sprintf(s, "./BreakReads/matchedCenters%d.txt", n);
    	fp1 = fopen(s,"w");

    	for(i=0;i<blcklngth;i++)
    	{
    		vchar = fgetc(fp);
			while(vchar != '\n')
			{
				putc(vchar,fp1);	
				vchar = fgetc(fp);		    		
			}
			putc('\n',fp1);
    	}
    	fclose(fp1);
    }

	sprintf(s, "./BreakReads/matchedCenters%d.txt", n);
	fp1 = fopen(s,"w");
    blcklngth = (NumReads - (np-1)*blcklngth);

	for(i=0;i<blcklngth;i++)
	{
		vchar = fgetc(fp);
		while(vchar != '\n')
		{
			putc(vchar,fp1);	
			vchar = fgetc(fp);		    		
		}
		putc('\n',fp1);
	}

	fclose(fp1);
	fclose(fp);



    blcklngth = (int) floor((NumReads+0.0)/np);
    fp = fopen("./reads","r");
    for (n=0;n<np-1;n++)
    {
    	sprintf(s, "./BreakReads/reads%d", n);
    	fp1 = fopen(s,"w");

    	for(i=0;i<blcklngth;i++)
    	{
    		vchar = fgetc(fp);
			while(vchar != '\n')
			{
				putc(vchar,fp1);	
				vchar = fgetc(fp);		    		
			}
			putc('\n',fp1);
    	}
    	fclose(fp1);
    }

	sprintf(s, "./BreakReads/reads%d", n);
	fp1 = fopen(s,"w");
    blcklngth = (NumReads - (np-1)*blcklngth);

	for(i=0;i<blcklngth;i++)
	{
		vchar = fgetc(fp);
		while(vchar != '\n')
		{
			putc(vchar,fp1);	
			vchar = fgetc(fp);		    		
		}
		putc('\n',fp1);
	}

	fclose(fp1);
	fclose(fp);
}

void breakReads(int np)
{
	FILE *fp, *fp1,*fpActLength, *fpActLength1;
	int NumReads,L,i,ell,temp,k,blcklngth,n,Length;
	char vchar;
	char s[100];
	fp = fopen("./dimensions.txt", "r");
	fscanf (fp, "%d", &NumReads);
	fscanf (fp, "%d", &Length);
	fclose(fp);
	/*
	*sparsity_power = 1.07;
	if (NumReads>30000)
		*sparsity_power=1.05;
	if (NumReads>40000)
		*sparsity_power=1.03;
	if (NumReads>60000)
		*sparsity_power=1.02;
	if (NumReads>100000)
		*sparsity_power=1.01;
	if (NumReads>300000)
		*sparsity_power=1.0;
*/


    blcklngth = (int) floor((NumReads+0.0)/np);

    
    fp = fopen("./ReadsNoisy.txt","r");
    fpActLength = fopen("./ActLength.txt","r");

    
    for (n=0;n<np-1;n++)
    {
    	sprintf(s, "./BreakReads/ReadsNoisy%d.txt", n);
    	fp1 = fopen(s,"w");
    	sprintf(s, "./BreakReads/ActLength%d.txt", n);
    	fpActLength1 = fopen(s,"w");

    	for(i=0;i<blcklngth;i++)
    	{
        	fscanf (fpActLength, "%d", &L);
			fprintf(fpActLength1,"%d\n",L);
    		for(ell=0;ell<Length;ell++)
    		{
    			fread((void*)(&vchar), sizeof(vchar), 1, fp);
    			fprintf(fp1,"%c",vchar);
    		}
    		fread((void*)(&vchar), sizeof(vchar), 1, fp);
			fprintf(fp1,"\n");
    	}
    	fclose(fp1);
    	fclose(fpActLength1);
    }


    blcklngth = (NumReads - (np-1)*blcklngth);
   	sprintf(s, "./BreakReads/ReadsNoisy%d.txt", np-1);
	fp1 = fopen(s,"w");

	sprintf(s, "./BreakReads/ActLength%d.txt", np-1);
	fpActLength1 = fopen(s,"w");


	for(i=0;i<blcklngth;i++)
	{
    	fscanf (fpActLength, "%d", &L);
		fprintf(fpActLength1,"%d\n",L);

		for(ell=0;ell<Length;ell++)
		{
			fread((void*)(&vchar), sizeof(vchar), 1, fp);
			fprintf(fp1,"%c",vchar);
		}
		fread((void*)(&vchar), sizeof(vchar), 1, fp);
		fprintf(fp1,"\n");
	}
	fclose(fp1);
	fclose(fp);
	fclose(fpActLength);
	fclose(fpActLength1);

}



void trimReads(char *InputFasta, int Reverse)
{
    int CutLength;
    FILE *fpreads, *fpCCSIsoSeq, *fpActLength, *fpdimension, *fpReadsNoisy,*fp1;
    if (Reverse == 0)
    {
	    fpCCSIsoSeq = fopen(InputFasta,"r");
	    fpreads = fopen("./reads","w");
	    fpActLength = fopen("./ActLength.txt","w");
	    Gen_reads_ActLen(&CutLength,fpCCSIsoSeq,fpreads,fpActLength);
	    fclose(fpCCSIsoSeq);
	    fclose(fpreads);
	    fclose(fpActLength);
	}
    if (Reverse == 1)
    {
	    fpCCSIsoSeq = fopen(InputFasta,"r");
	    fpreads = fopen("./readsOrig","w");
	    fpActLength = fopen("./ActLength.txt","w");
	    Gen_reads_ActLen(&CutLength,fpCCSIsoSeq,fpreads,fpActLength);
	    fclose(fpCCSIsoSeq);
	    fclose(fpreads);
	    fclose(fpActLength);

	    fp1 = fopen("./readsOrig","r");
	    fpreads = fopen("./reads","w");
	    ReverseReads(fp1,fpreads);
	    fclose(fp1);
	    fclose(fpreads);
	}

    fpreads = fopen("./reads","r");
    fpReadsNoisy = fopen("./ReadsNoisy.txt","w");
    fpdimension = fopen("./dimensions.txt","w");
    Gen_dim_Len_ReadsNoisy(CutLength, fpreads,fpReadsNoisy, fpdimension);
    fclose(fpreads);
    fclose(fpReadsNoisy);
    fclose(fpdimension);
    remove("./readsOrig");
}
/*
void CharToBool(char v, bool *b1, bool *b2)
{
	if (v=='A')
	{
		*b1 = 0;
		*b2 = 0;
	}
	if (v=='C')
	{
		*b1 = 0;
		*b2 = 1;		
	}
	if (v=='G')
	{
		*b1 = 1;
		*b2 = 0;		
	}
	if (v=='T')
	{
		*b1 = 1;
		*b2 = 1;		
	}
}
*/
void BoolToChar(char *v, bool b1, bool b2)
{
	if ((b1 ==0)&&(b2 ==0))
		*v = 0;
	if ((b1 ==0)&&(b2 ==1))
		*v = 1;
	if ((b1 ==1)&&(b2 ==0))
		*v = 2;
	if ((b1 ==1)&&(b2 ==1))
		*v = 3;
}

void ReverseReads(FILE *fpOrig,FILE *fpReverse)
{

    int  ell;
    char v,Seq[50000];


    v = fgetc(fpOrig);
    while (v != EOF) 
    {
        ell = 0;

        Seq[ell]= v;
        while(Seq[ell] != '\n')
        {
            ell++;
            Seq[ell] = fgetc(fpOrig);
        }
        ell--;
        while(ell>=0)
        {
            putc(Seq[ell],fpReverse);
            ell--;
        }
        putc('\n',fpReverse);
        v = fgetc(fpOrig);
    }


}



void Gen_dim_Len_ReadsNoisy(int CutLength,FILE *fpreads,FILE *fpReadsNoisy,FILE *fpdimension)
{
    int i,k;


    char v;
    v = fgetc(fpreads);
    int j = 0;
    while (v != EOF)
    {
        i = 0;
        j++;
        while ((v != '\n') && (v != EOF))
        {
            if (i < CutLength)
            {
                putc(v,fpReadsNoisy);
                i++;
            }
            v = fgetc(fpreads);
        }

      
        while(i < CutLength)
        {
            putc('A',fpReadsNoisy);
            i++;
        }
        fprintf(fpReadsNoisy, "\n");
        v = fgetc(fpreads);
    }
    fprintf(fpdimension,"%d\n",j);


    fprintf(fpdimension,"%d\n",CutLength);
}

void Gen_reads_ActLen(int *CutLength,FILE *fpCCSIsoSeq,FILE *fpreads,FILE *fpActLength)
{
    int *ActLengthArr,i; 
    ActLengthArr = (int*)calloc(1000000, sizeof(int));
    char v;
    v = fgetc(fpCCSIsoSeq);
    int counter;
    counter = 0;
    while(v !=EOF)
    {
        if (v == '>')
        {
            while(v != '\n')
            {
                v = fgetc(fpCCSIsoSeq);
            }
            v = fgetc(fpCCSIsoSeq);
        }
        i = 0;
        while((v != '>') && (v != EOF))
        {
            if(v!= '\n')
            {
            	if ((v > 43) && (v < 123))
                	putc(v,fpreads);
                i++;
            }
            v = fgetc(fpCCSIsoSeq);
        }
        putc('\n',fpreads);
        fprintf(fpActLength,"%d\n",i);


        ActLengthArr[counter] = i;
        counter++;
    }
    quickSort( ActLengthArr, 0, counter-1);
    *CutLength = ActLengthArr[((int) ceil((counter-1.0)*0.995))] + 400;
    free(ActLengthArr);
}


void quickSort( int a[], int l, int r)
{
   int j;

   if( l < r ) 
   {
   	// divide and conquer
        j = partition( a, l, r);
       quickSort( a, l, j-1);
       quickSort( a, j+1, r);
   }
	
}

int partition( int a[], int l, int r) {
   int pivot, i, j, t;
   pivot = a[l];
   i = l; j = r+1;
		
   while( 1)
   {
   	do ++i; while( a[i] <= pivot && i <= r );
   	do --j; while( a[j] > pivot );
   	if( i >= j ) break;
   	t = a[i]; a[i] = a[j]; a[j] = t;
   }
   t = a[l]; a[l] = a[j]; a[j] = t;
   return j;
}


void ClusteringInfo()
{
	int M,tempInt,MaxClusterSize,NumReads,i,j,r,tempIdx,tempInt2;
	char vchar,s[100];
	float tempFloat;
	FILE *fp1, *fp2, *fp3;

	fp1 = fopen("FinalCenters.txt","r");
   	vchar = fgetc(fp1);
   	M = 0;
   	while (vchar != EOF)
   	{
   		if (vchar == '\n')
   			M++;
   		vchar = fgetc(fp1);
   	}
   	fclose(fp1);

   	int ClusterSizeAry[M];
	 
   	fp1 = fopen("ClusterSizes.txt","r");
   	MaxClusterSize = 0;
   	for(i=0;i<M;i++)
   	{
		fscanf (fp1, "%d", &tempInt);
		ClusterSizeAry[i] = tempInt;
		if (tempInt>MaxClusterSize)
			MaxClusterSize = tempInt;
   	}
   	fclose(fp1);

    int *ClusterReadsIdAry, *ClusterEndAlignAry,IdxAry[M]; 
    ClusterReadsIdAry = (int*)calloc(M * MaxClusterSize, sizeof(int));
    ClusterEndAlignAry = (int*)calloc(M * MaxClusterSize, sizeof(int));

    double *ClusterScoreAry;
    ClusterScoreAry = (double*)calloc(M * MaxClusterSize, sizeof(double));


	fp1 = fopen("./dimensions.txt", "r");
	fscanf (fp1, "%d", &NumReads);
	fclose(fp1);

	for (i=0;i<M;i++)
		IdxAry[i]=0;


	fp1 = fopen("CentroidId.txt","r");
	fp2 = fopen("EndAlign.txt","r");
	fp3 = fopen("NormalScores.txt","r");
	for (r=0;r<NumReads;r++)
	{

		fscanf(fp2, "%d", &tempInt2);
		fscanf(fp3, "%f", &tempFloat);

		fgets(s, sizeof(s), fp1);
		if (sscanf(s,"%d",&tempInt) == 1)
		{
			tempIdx = IdxAry[tempInt]+ tempInt * MaxClusterSize;
			IdxAry[tempInt] = IdxAry[tempInt] + 1;
			ClusterReadsIdAry[tempIdx] = r;

			ClusterEndAlignAry[tempIdx] = tempInt2;
			ClusterScoreAry[tempIdx] = tempFloat;
		}

	}
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);


	fp1 = fopen("ClusterReadsId.txt","w");
	fp2 = fopen("ClusterEndAlign.txt","w");
	fp3 = fopen("ClusterNormalScore.txt","w");
	for (i=0;i<M;i++)
	{
		for(j=0;j<ClusterSizeAry[i];j++)
		{
			fprintf(fp1,"%d\n",ClusterReadsIdAry[j+ i * MaxClusterSize]);
			fprintf(fp2,"%d\n",ClusterEndAlignAry[j+ i * MaxClusterSize]);
			fprintf(fp3,"%.3f\n",ClusterScoreAry[j+ i * MaxClusterSize]);
		}
		
	}
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);
	free(ClusterReadsIdAry);
	free(ClusterEndAlignAry);
	free(ClusterScoreAry);
}

void DetectCentroidLength()
{
	FILE *fp1,*fp2,*fp3;
	int i,M,j,tempInt, a[20000];
	char vchar;
 
	fp1 = fopen("FinalCenters.txt","r");
   	vchar = fgetc(fp1);
   	M = 0;
   	while (vchar != EOF)
   	{
   		if (vchar == '\n')
   			M++;
   		vchar = fgetc(fp1);
   	}
   	fclose(fp1);

   	fp1 = fopen("ClusterSizes.txt","r");
   	fp2 = fopen("ClusterEndAlign.txt","r");
   	fp3 = fopen("CentroidLength.txt","w");

   	for (i=0; i<M; i++)
   	{
		fscanf(fp1, "%d", &tempInt);
		for (j=0; j< tempInt; j++)
		{		
			fscanf(fp2, "%d", &a[j]);
		}
		fprintf(fp3,"%d\n",almostmedian(a,tempInt));
   	}
   	fclose(fp1);
   	fclose(fp2);
   	fclose(fp3);
}


int almostmedian(int x[], int n) 
{
    int temp;
    int i, j;
    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                // swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }
    if(n%2==0) {
        // if there is an even number of elements, return the smaller (not median)
        //return((x[n/2] + x[n/2 - 1]) / 2.0);
        return(x[n/2]);
    } else {
        // else return the element in the middle
        return x[n/2];
    }
}


void AdjustCentroidLength()
{
        FILE *fp1,*fp2,*fp3;
        int i,M,j,tempInt;
        char vchar;

        fp1 = fopen("FinalCenters.txt","r");
        vchar = fgetc(fp1);
        M = 0;
        while (vchar != EOF)
        {
                if (vchar == '\n')
                        M++;
                vchar = fgetc(fp1);
        }
        fclose(fp1);

        fp1 = fopen("CentroidLength.txt","r");
        fp2 = fopen("FinalCenters.txt","r");
        fp3 = fopen("FinalCentersTrimmed.txt","w");

        for (i=0;i<M; i++)
        {
                fscanf(fp1, "%d", &tempInt);
                for (j=0;j<tempInt; j++)
                {
                        vchar = fgetc(fp2);
                        if (vchar == '\n')
                        {
                                break;
                        }
                        putc(vchar,fp3);
                }
                while(vchar !='\n')
                        vchar = fgetc(fp2);
                putc('\n',fp3);
        }
        fclose(fp1);
        fclose(fp2);
        fclose(fp3);
}



double ScoreCalTotRecFast(int LOrig, int LRead, int *Orig, int *Read, double InsScore, double DelScore, double MutScore, double CorrScore)
{
	int LcutOrig = 9;
	int jIdx = 0;
	int LcutRead = 9;
	double Score,tempdouble,RelInitScore;
	int i,ReljIdx,LReadTemp;
	ReljIdx = -1;
	tempdouble = 0;
	for (i=0;i<LOrig - LcutOrig+1; i++)
	{
		LReadTemp = LcutRead;
		if ((LRead - ReljIdx) <LcutRead)
			LReadTemp = LRead - ReljIdx;
		jIdx = jIdx + ReljIdx + 1;
		if (jIdx<LRead)
		{
			Score = ScoreCalRecFast(&RelInitScore,&ReljIdx, LcutOrig,LReadTemp, &Orig[i], &Read[jIdx], InsScore, DelScore, MutScore, CorrScore);
			tempdouble += RelInitScore;
		}
		if (jIdx>=LRead)
		{
			jIdx = LRead+1;
			tempdouble += DelScore;
		}

		// if not aligned initially, then break
		if (tempdouble<-20)
			break;

	}
	if (jIdx < LRead+1)
	{
		tempdouble -= RelInitScore;
		Score += tempdouble;
	}
	if (jIdx == LRead+1)
		Score = tempdouble +  DelScore * (LcutOrig-1);
	return(Score);
}

double ScoreCalRecFast(double *RelInitScore, int *ReljIdx, int LOrig, int LRead, int *Orig, int *Read, double InsScore, double DelScore, double MutScore, double CorrScore)
{
	double Score[LOrig+1][LRead+1]; //add one row and column full of zero
	// Ins 0; Del 1; Corr 2; Mut 3;
	int DynamicPath[LOrig+1][LRead+1]; 
	int i,ii,jj,j,k,jmax,tempint,TempAligning[3*LOrig];
	double temp;
	
	
	for (i=0;i<=LOrig;i++)
	{
		Score[i][0] = i * DelScore;
		DynamicPath[i][0] = 1;
	}
	for (j=0;j<=LRead;j++)
	{
		Score[0][j] = j * InsScore;
		DynamicPath[0][j] = 0;
	}
	
	for (i=1; i<=LOrig;i++)
		for (j=1;j<=LRead;j++)
		{
			temp = Score[i][j-1] + InsScore;
			DynamicPath[i][j] = 0;
			if ((Score[i-1][j] + DelScore) > temp)
			{
				temp = Score[i-1][j] + DelScore;
				DynamicPath[i][j] = 1;
			}
			if (Orig[i-1] == Read[j-1])
			{
				if ((Score[i-1][j-1] + CorrScore) > temp)
				{
					temp = (Score[i-1][j-1] + CorrScore);
					DynamicPath[i][j] = 2;
				}
			}
			else
			{
				if ((Score[i-1][j-1] + MutScore) > temp)
				{
					temp = (Score[i-1][j-1] + MutScore);
					DynamicPath[i][j] = 3;
				}
			}		
			Score[i][j] = temp;
		}
	jmax = LRead;
	for (j=0;j<=LRead;j++)
	{
		if(Score[LOrig][j] > temp)
		{
			temp = Score[LOrig][j];
			jmax = j;
		}
	}
	k=0;
	i = LOrig;
	j = jmax;
	while((i>0)||(j>0))
	{
		TempAligning[k] = DynamicPath[i][j]; 
		if (DynamicPath[i][j] == 0)
		{
			jj = j-1;
			ii = i;
		}
		if (DynamicPath[i][j] == 1)
		{
			ii = i-1;
			jj = j;
		}
		if (DynamicPath[i][j] == 2)
		{
			jj = j-1;
			ii = i-1;
		}
		if (DynamicPath[i][j] == 3)
		{
			ii = i-1;
			jj = j-1;
		}	
		j = jj;
		i = ii;
		k +=1;
	}
	for (i=0;i<(k-1)/2;i++)
	{	
		tempint = TempAligning[k-i-1];
		TempAligning[k - i-1] = TempAligning[i];
		TempAligning[i] = tempint;
	} 
	i = 0;
	while((TempAligning[i]==0)&&(i<k))
	{
		i += 1;
	}
	*ReljIdx = i; 
	*RelInitScore = i * InsScore;
	if (TempAligning[i] == 1)
	{
		*ReljIdx = *ReljIdx -1 ;
		*RelInitScore = *RelInitScore + DelScore;
	}
	if (TempAligning[i] == 2)
	{
		*RelInitScore = *RelInitScore + CorrScore;
	}
	if (TempAligning[i] == 3)
	{
		*RelInitScore = *RelInitScore + MutScore;
	}
	
	
	return(temp);
}


