#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#include <assert.h>


void ComputeAdjacency(int M, bool *Adjacency);
double ScoreCalTotRecFast(int LOrig, int LRead, int *Orig, int *Read, double InsScore, double DelScore, double MutScore, double CorrScore);
double ScoreCalRecFast(double *RelInitScore, int *ReljIdx, int LOrig, int LRead, int *Orig, int *Read, double InsScore, double DelScore, double MutScore, double CorrScore);
void DFS(bool *Adjacency, int M, int i, bool *visited);
void FindConnectedComp(int M,int *CompID,bool *Adjacency);
void FindNewCentroidId(int M, int *CompID, int *NewCentroidId, int *NewClusterSize, bool *CentroidFiltering);
void ExtractHQCentroids(bool *CentroidFiltering);

int const FilterClusterSize = 2;

int main(int argc, char **argv)
{	
	chdir("Clusters");
	chdir(argv[1]);
	FILE *fp;
	int M,i,j,tempInt;
	char vchar;
	fp = fopen("ClusterSizes.txt","r");
   	vchar = fgetc(fp);
   	M = 0;
   	while (vchar != EOF)
   	{
   		if (vchar == '\n')
   			M++; // Number of detected clusters
   		vchar = fgetc(fp);
   	}
   	fclose(fp);

	bool *Adjacency;
    	Adjacency = (bool*)calloc(M * M, sizeof(bool));
	
	// MxM matrix for distance of the detected centroids 
    	for(i=0;i<M;i++)
    		for(j=0;j<M;j++)
    			Adjacency[i+M*j] = 0;

	ComputeAdjacency(M, Adjacency);

	for(i=0;i<M;i++)
		for(j=0;j<M;j++)
			if (Adjacency[i + M*j])
				Adjacency[j + M*i] = 1;


	int *CompID;
	CompID = (int*)calloc(M,sizeof(int));

	FindConnectedComp(M, CompID, Adjacency);

	int *NewCentroidId;
	NewCentroidId = (int*)calloc(M,sizeof(int));
	int *NewClusterSize;
	NewClusterSize = (int*)calloc(M,sizeof(int));
	bool *CentroidFiltering;
	CentroidFiltering = (bool*)calloc(M,sizeof(int));

	FindNewCentroidId(M, CompID, NewCentroidId, NewClusterSize,CentroidFiltering);


	ExtractHQCentroids(CentroidFiltering);


	fp = fopen("./Filtering","w");
	for(i=0;i<M;i++)
		fprintf(fp, "%d\n", CentroidFiltering[i]);
	fclose(fp);



	free(Adjacency);
	free(CompID);
	free(NewCentroidId);
	return 0;
}

void ExtractHQCentroids(bool *CentroidFiltering)
{
	FILE *fp1,*fp2;
	char vchar;
	int counter;
	fp1 = fopen("./FinalCentersTrimmedRev.txt","r");
	fp2 = fopen("./ColapsedCentroid.txt","w");
	vchar = fgetc(fp1);
	counter = 0;

	while(vchar!=EOF)
	{
		while(vchar != '\n')
		{
			if (CentroidFiltering[counter] == 1)
				putc(vchar,fp2);
			vchar = fgetc(fp1);
		}
		if (CentroidFiltering[counter] == 1)
			putc('\n',fp2);
		counter++;
		vchar = fgetc(fp1);
	}

	fclose(fp1);
	fclose(fp2);
}

void FindNewCentroidId(int M, int *CompID, int *NewCentroidId, int *NewClusterSize, bool *CentroidFiltering)
{
    FILE *fp, *fp2;
    int i,tempInt,NumComp,j;

    fp = fopen("./ClusterSizes.txt","r");
    int *ClusterSize;
    ClusterSize = (int*)calloc(M, sizeof(int));
    for (i=0;i<M;i++)
    {
        fscanf(fp, "%d", &tempInt);
        ClusterSize[i] = tempInt;
    }
    fclose(fp);

    NumComp = 0;
    for (i=0;i<M;i++)
    	if (CompID[i] > NumComp)
    		NumComp = CompID[i];
    NumComp++;

    int *IdxMaxSize;
    IdxMaxSize = (int*)calloc(NumComp, sizeof(int));

    int *CompCoverage;
    CompCoverage = (int*)calloc(NumComp, sizeof(int));
    for (i=0;i<NumComp;i++)
    {
    	CompCoverage[i]=0;
    	for(j=0;j<M; j++)
	    	if (CompID[j] == i)
    			CompCoverage[i] += ClusterSize[j];
    }


    for (i=0;i<NumComp;i++)
    {
    	tempInt = -1;
    	for (j=0; j<M; j++)
	    	if ((CompID[j] == i)&&(ClusterSize[j]>tempInt))
    		{
    			IdxMaxSize[i] = j;
    			tempInt = ClusterSize[j];
	    	}
    }

    for (i=0;i<M;i++)
    {
    	NewCentroidId[i] = IdxMaxSize[CompID[i]];
    	NewClusterSize[i] = CompCoverage[CompID[i]];
    }

    fp = fopen("./IMS","w");
    for (i=0;i<NumComp;i++)
    	fprintf(fp,"%d\n",IdxMaxSize[i]);
    fclose(fp);

    for (i=0;i<M;i++)
    	CentroidFiltering[i] = 0;

    for (i=0; i<NumComp;i++)
    	if (CompCoverage[i]>FilterClusterSize)
    		CentroidFiltering[IdxMaxSize[i]] = 1;

    
    // Update centroid ids
    int N = 0;
    char vchar;
    fp = fopen("./CentroidId.txt","r");
    vchar = fgetc(fp);
    while (vchar != EOF) {
	if (vchar == '\n')
		N++;
   	vchar = fgetc(fp);
	}
    fclose(fp);
    printf("Number of reads: %d\n", N);

    fp = fopen("./CentroidId.txt","r");
    fp2 = fopen("./FinalCentroidId.txt","w");
    int *ReadIds;
    ReadIds = (int*)calloc(N, sizeof(int));
    for (i=0;i<N;i++)
    {
        fscanf(fp, "%d", &tempInt);
        ReadIds[i] = NewCentroidId[tempInt];
	// printf("Cluster id[%d]: %d\n", i, ReadIds[i]);
	fprintf(fp2,"%d\n", ReadIds[i]);
    }
    fclose(fp);
    fclose(fp2);

    free(IdxMaxSize);
    free(ClusterSize);
    free(CompCoverage);
    free(ReadIds);
}



void FindConnectedComp(int M,int *CompID,bool *Adjacency)
{
	int i,counter,j;
	bool *Allvisited;
	Allvisited = (bool*)calloc(M,sizeof(bool));


	bool *ComponentVisit;
	ComponentVisit = (bool*)calloc(M,sizeof(bool));

	for (i=0;i<M;i++)
		Allvisited[i]=0;

	i = 0;
	counter = 0;

	while(i<M)
	{
		if (!Allvisited[i])
		{
			for (j=0;j<M;j++)
				ComponentVisit[j]=0;

			DFS(Adjacency,M,i,ComponentVisit);
			for (j=0;j<M;j++)
				if (ComponentVisit[j]) 
				{
					CompID[j] = counter;
					Allvisited[j] = 1;
				}
			counter++;
		}
		i++;
	}
	free(Allvisited);
	free(ComponentVisit);
}

 
 
void DFS(bool *Adjacency, int M, int i, bool *visited)
{
    int j;
    visited[i]=1;
    
    for(j=0;j<M;j++)
       if(!visited[j] && Adjacency[i+M*j]==1)
            DFS(Adjacency,M, j, visited);
}



void ComputeAdjacency(int M, bool *Adjacency)
{
	FILE *fp,*fp1;
	char vchar;
	int tempInt,i,j;



	fp = fopen("./CentroidLength.txt","r");
	int *CentroidLength;
    	CentroidLength = (int*)calloc(M, sizeof(int));
    	for (i=0;i<M;i++)
    	{
        	fscanf(fp, "%d", &tempInt);
        	CentroidLength[i] = tempInt;
    	}
    	fclose(fp);

	fp = fopen("./ClusterSizes.txt","r");
	int *ClusterSize;
    	ClusterSize = (int*)calloc(M, sizeof(int));
    	for (i=0;i<M;i++)
    	{
		fscanf(fp, "%d", &tempInt);
       	 	ClusterSize[i] = tempInt;
    	}
    	fclose(fp);

	int ell,LOrig,LRead;
	int IgnoreEndSize = 15;
	int IgnoreBeginingSize = 15;

	FILE *fpDist;
	fpDist = fopen("./dist","w");

        int *Y;
        Y = (int*)calloc(30000, sizeof(int));
    	int *S;
    	S = (int*)calloc(30000, sizeof(int));

 	double CorrScore, MutScore,DelScore,InsScore,temp1,temp2;
 	CorrScore = 0.0;
 	MutScore = -1.0;
 	DelScore = -1.0;
 	InsScore = -1.0;

	fp = fopen("FinalCentersTrimmed.txt","r");
	char v1,v2;
	v1 = fgetc(fp);
	i = 0;
	while (v1 != EOF)
	{
		if(i%100 ==0)
			printf("i = %d\n",i);
		ell = 0;
    	while(v1 != '\n')
		{
	        if (v1=='A')
	        	S[ell] = 0;
	        else if (v1 == 'C')
	        	S[ell] = 1;
			else if (v1 == 'G')
				S[ell] = 2;
			else 
				S[ell] = 3;
			v1 = fgetc(fp);
	    	ell++;
	    }
	    LOrig = ell;
	    v1 = fgetc(fp);

		fp1 = fopen("FinalCentersTrimmed.txt","r");
	    v2 = fgetc(fp1);
	    j = 0;
	    while(v2 != EOF)
	    {
	    	Adjacency[i + M*j] = 0;
	    	ell = 0;
	    	while(v2 != '\n')
	    	{
		        if (v2=='A')
		        	Y[ell] = 0;
		        else if (v2 == 'C')
		        	Y[ell] = 1;
				else if (v2 == 'G')
					Y[ell] = 2;
				else 
					Y[ell] = 3;
				v2 = fgetc(fp1);
		    	ell++;
	    	}
	    	v2 = fgetc(fp1);
	    	LRead = ell;

	    	if ((abs(CentroidLength[i]-CentroidLength[j])<15)&&(ClusterSize[i]>FilterClusterSize)&&(ClusterSize[j]>FilterClusterSize))
	    	{

		    	if (IgnoreEndSize<LOrig)
		    	{
			    	temp1 = ScoreCalTotRecFast(LOrig-IgnoreEndSize,LRead, S, Y, InsScore, DelScore, MutScore, CorrScore);
		        	temp1 -= ScoreCalTotRecFast(IgnoreBeginingSize,LRead, S, Y, InsScore, DelScore, MutScore, CorrScore);
		    	}
		    	else
		    		temp1 = -100;
//		    	if (temp1>-0.005 * (CentroidLength[i]+0.0))
		    	if ((temp1>-6)||(temp1>-0.005*CentroidLength[i]))
		    	{
		    		Adjacency[i+M*j] = 1;
			    	fprintf(fpDist, "%d  ", 1);
		    	}
		    	else
		    	{
			    	fprintf(fpDist, "%d  ", 0);
		    	}
	    	}
	    	else
	    	{
		    	fprintf(fpDist, "%d  ", 0);
	    	}

		    j++;
	    }
	    fprintf(fpDist,"\n");
	    fclose(fp1);
		i++;

	}
	fclose(fp);
	fclose(fpDist);
	free(Y);
	free(S);
}


double ScoreCalTotRecFast(int LOrig, int LRead, int *Orig, int *Read, double InsScore, double DelScore, double MutScore, double CorrScore)
{
	int LcutOrig = 10;
	int jIdx = 0;
	int LcutRead = 10;
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
		if (tempdouble<-25)
		{
			tempdouble = -100;
			break;
		}


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


