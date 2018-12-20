#include "mex.h"
#include <stdio.h>
#include <math.h>
#include <string.h>


/*
 * -------------------------------------------------------------------
 *                       Function and parameter
 * -------------------------------------------------------------------
 *
 * altra solves the following problem
 *
 * 1/2 \|x-v\|^2 + \sum \lambda_i \|x_{G_i}\|,
 *
 * where x and v are of dimension n,
 *       \lambda_i >=0, and G_i's follow the tree structure
 *
 * The file is implemented in the following in Matlab:
 *
 * x=altra(v, n, ind, nodes);
 *
 * ind is a 3 x nodes matrix.
 *       Each column corresponds to a node.
 *
 *       The first element of each column is the starting index,
 *       the second element of each column is the ending index
 *       the third element of each column corrreponds to \lambbda_i.
 *
 * -------------------------------------------------------------------
 *                       Notices:
 * -------------------------------------------------------------------
 *
 * 1. The nodes in the parameter "ind" should be given in the 
 *    either
 *           the postordering of depth-first traversal
 *    or 
 *           the reverse breadth-first traversal.
 *
 * 2. When each elements of x are penalized via the same L1 
 *    (equivalent to the L2 norm) parameter, one can simplify the input
 *    by specifying 
 *           the "first" column of ind as (-1, -1, lambda)
 *
 *    In this case, we treat it as a single "super" node. Thus in the value
 *    nodes, we only count it once.
 *
 * 3. The values in "ind" are in [1,n].
 *
 * 4. The third element of each column should be positive. The program does
 *    not check the validity of the parameter. 
 *
 *    It is still valid to use the zero regularization parameter.
 *    In this case, the program does not change the values of 
 *    correponding indices.
 *    
 *
 * -------------------------------------------------------------------
 *                       History:
 * -------------------------------------------------------------------
 *
 * Composed by Jun Liu on April 20, 2010
 *
 * For any question or suggestion, please email j.liu@asu.edu.
 *
 */
 
void altra_yj(double *x, double *v, int n, double *ind, int nodes, double *lambda_vector){
    
    int i, j, m; // �������i,j,m
    double lambda,twoNorm, ratio; // �������lambda, twoNorm, ratio
    
    /*
     * test whether the first node is special �жϵ�һ���ڵ��ǲ���special��
     */
    if ((int) ind[0]==-1){ //�ж�ind(1,1)�ǲ���==-1
        
        /*
         *Recheck whether ind[1] equals to zero
         */
        if ((int) ind[1]!=-1){ //�ж�ind(1,2)�ǲ���==-1,��������ڣ����򱨴�
            printf("\n Error! \n Check ind");
            exit(1);
        }        
        
       
        
        for(j=0;j<n;j++){ //ÿ��Ԫ�ض��Ƚ���v��lambda��ֵ�����v����lambda ��x=v����lambda�����vС��lambda��x=v����lambda���������x=0
            if (v[j]>lambda_vector[j])
                x[j]=v[j]-lambda_vector[j];
            else
                if (v[j]<-lambda_vector[j])
                    x[j]=v[j]+lambda_vector[j];
                else
                    x[j]=0;
        }
        
        i=1; //���ind[0]=-1;i=1;����i=0
    }
    else{
        memcpy(x, v, sizeof(double) * n);
        i=0;
    }
            
    /*
     * sequentially process each node
     *
     */
	for(;i < nodes; i++){ //Ҫôi��0��number of groups��Ҫôi��1��number of groups
        /*
         * compute the L2 norm of this group ����������L2����
         */
		twoNorm=0;//��ʼ��������Ϊ0
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)//��ʼforѭ�������i=0,��ôj��-1��ʼ��һֱ��1
            //���i=1,j��ind[3]-1��ʼ(�ȼ���ind(1,1))��һֱ��ind[4](�ȼ���ind(1,2))(�൱�ڿ�ʼѭ����һ�������)
			twoNorm += x[j] * x[j];        
        twoNorm=sqrt(twoNorm); //����������x�ľ���ֵ
        
        lambda=ind[3*i+2];//lambda=ind[5](�ȼ���ind(1,3))
        if (twoNorm>lambda){ //�����ǰԪ�صĶ�����ֵ����lambda
            ratio=(twoNorm-lambda)/twoNorm;//ratio=��������-lambda��/������
            
            /*
             * shrinkage this group by ratio�������ֵ���ձ���ratioѹ��
             */
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)//��ʼѭ����j��ind[3]-1��ʼ(�ȼ���ind(1,1))��һֱ��ind[4](�ȼ���ind(1,2))(�൱�ڿ�ʼѭ����һ�������)
                x[j]*=ratio; //����ǰ��ֵ����ratio           
        }
        else{
            /*
             * threshold this group to zero ���������ֵС��lambda ������=0
             */ 
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[j]=0;
        }
	}
}

void altra_mt_yj(double *X, double *V, int n, int k, double *ind, int nodes, double *Lambda_vector){
    int i, j;
    
    double *x=(double *)malloc(sizeof(double)*k);
    double *v=(double *)malloc(sizeof(double)*k);
    double *lambda_vector=(double *)malloc(sizeof(double)*k);
    
    for (i=0;i<n;i++){
        /*
         * copy a row of V to v
         *         
         */
        for(j=0;j<k;j++)
            v[j]=V[j*n + i];
            lambda_vector[j]=Lambda_vector[j*n + i];
        
        altra_yj(x, v, k, ind, nodes, lambda_vector);
        
        /*
         * copy the solution to X         
         */        
        for(j=0;j<k;j++)
            X[j*n+i]=x[j];
    }
    
    free(x);
    free(v);
}

void treeNorm_yj(double *tree_norm, double *x, int n, double *ind, int nodes, double *lambda_vector){
    
    int i, j, m;
    double twoNorm, lambda;
    
    *tree_norm=0;
    
    /*
     * test whether the first node is special
     */
    if ((int) ind[0]==-1){
        
        /*
         *Recheck whether ind[1] equals to zero
         */
        if ((int) ind[1]!=-1){
            printf("\n Error! \n Check ind");
            exit(1);
        }        
        
       // lambda=ind[2];
        
        for(j=0;j<n;j++){
            *tree_norm+=fabs(x[j]) * lambda_vector[j];
        }
        
        //*tree_norm=*tree_norm * lambda;
        
        i=1;
    }
    else{
        i=0;
    }
            
    /*
     * sequentially process each node
     *
     */
	for(;i < nodes; i++){
        /*
         * compute the L2 norm of this group         
         */
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];        
        twoNorm=sqrt(twoNorm);
        
        lambda=ind[3*i+2];//�˴���LambdaӦ����group��
        
        *tree_norm=*tree_norm + lambda*twoNorm;
	}
}

