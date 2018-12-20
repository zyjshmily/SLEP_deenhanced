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
    
    int i, j, m; // 定义变量i,j,m
    double lambda,twoNorm, ratio; // 定义变量lambda, twoNorm, ratio
    
    /*
     * test whether the first node is special 判断第一个节点是不是special的
     */
    if ((int) ind[0]==-1){ //判断ind(1,1)是不是==-1
        
        /*
         *Recheck whether ind[1] equals to zero
         */
        if ((int) ind[1]!=-1){ //判断ind(1,2)是不是==-1,如果不等于，程序报错
            printf("\n Error! \n Check ind");
            exit(1);
        }        
        
       
        
        for(j=0;j<n;j++){ //每个元素都比较下v和lambda的值，如果v大于lambda 则x=v减掉lambda，如果v小于lambda则x=v加上lambda，如果等于x=0
            if (v[j]>lambda_vector[j])
                x[j]=v[j]-lambda_vector[j];
            else
                if (v[j]<-lambda_vector[j])
                    x[j]=v[j]+lambda_vector[j];
                else
                    x[j]=0;
        }
        
        i=1; //如果ind[0]=-1;i=1;否则i=0
    }
    else{
        memcpy(x, v, sizeof(double) * n);
        i=0;
    }
            
    /*
     * sequentially process each node
     *
     */
	for(;i < nodes; i++){ //要么i从0到number of groups，要么i从1到number of groups
        /*
         * compute the L2 norm of this group 计算这个组的L2范数
         */
		twoNorm=0;//初始化二范数为0
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)//开始for循环，如果i=0,那么j从-1开始，一直到1
            //如果i=1,j从ind[3]-1开始(等价于ind(1,1))，一直到ind[4](等价于ind(1,2))(相当于开始循环第一组的索引)
			twoNorm += x[j] * x[j];        
        twoNorm=sqrt(twoNorm); //二范数等于x的均方值
        
        lambda=ind[3*i+2];//lambda=ind[5](等价于ind(1,3))
        if (twoNorm>lambda){ //如果当前元素的二范数值大于lambda
            ratio=(twoNorm-lambda)/twoNorm;//ratio=（二范数-lambda）/二范数
            
            /*
             * shrinkage this group by ratio将这组的值按照比例ratio压缩
             */
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)//开始循环，j从ind[3]-1开始(等价于ind(1,1))，一直到ind[4](等价于ind(1,2))(相当于开始循环第一组的索引)
                x[j]*=ratio; //将当前的值乘上ratio           
        }
        else{
            /*
             * threshold this group to zero 如果二范数值小于lambda 则让其=0
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
        
        lambda=ind[3*i+2];//此处的Lambda应该是group的
        
        *tree_norm=*tree_norm + lambda*twoNorm;
	}
}

