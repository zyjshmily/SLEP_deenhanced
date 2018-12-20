#include "mex.h"
#include <stdio.h>
#include <math.h>
#include <string.h>


/*
 * Important Notice: September 20, 2010
 *
 * In this head file, we deal with the case that the features might not be well ordered.
 * 
 * If the features in the tree strucutre are well ordered, i.e., the indices of the left nodes is always less
 * than the right nodes, please refer to "altra.h".
 *
 * The advantage of "altra.h" is that, we donot need to use an explicit
 * variable for recording the indices.
 *
 *
 */

/*
 * -------------------------------------------------------------------
 *                       Functions and parameter
 * -------------------------------------------------------------------
 *
 * general_altra solves the following problem
 *
 * 1/2 \|x-v\|^2 + \sum \lambda_i \|x_{G_i}\|,
 *
 * where x and v are of dimension n,
 *       \lambda_i >=0, and G_i's follow the tree structure
 *
 * It is implemented in Matlab as follows:
 *
 * x=general_altra(v, n, G, ind, nodes);
 *
 * G contains the indices of the groups.
 *   It is a row vector. Its length equals to \sum_i \|G_i\|.
 *   If all the entries are penalized with L1 norm,
 *      its length is \sum_i \|G_i\| - n.
 *
 * ind is a 3 x nodes matrix.
 *       Each column corresponds to a node.
 *
 *       The first element of each column is the starting index,
 *       the second element of each column is the ending index
 *       the third element of each column corrreponds to \lambbda_i.
 *
 *
 *
 * The following example shows how G and ind works:
 *
 * G={ {1, 2}, {4, 5}, {3, 6}, {7, 8},
 *     {1, 2, 3, 6}, {4, 5, 7, 8}, 
 *     {1, 2, 3, 4, 5, 6, 7, 8} }.
 *
 * ind={ [1, 2, 100]', [3, 4, 100]', [5, 6, 100]', [7, 8, 100]',
 *       [9, 12, 100]', [13, 16, 100]', [17, 24, 100]' }
 * 
 * where "100" denotes the weight for the nodes.
 *
 *
 *
 * -------------------------------------------------------------------
 *                       Notices:
 * -------------------------------------------------------------------
 *
 * 1. The features in the tree might not be well ordered. Otherwise, you are
 *    suggested to use "altra.h".
 *
 * 2. When each elements of x are penalized via the same L1 
 *    (equivalent to the L2 norm) parameter, one can simplify the input
 *    by specifying 
 *           the "first" column of ind as (-1, -1, lambda)
 *
 *    In this case, we treat it as a single "super" node. Thus in the value
 *    nodes, we only count it once.
 *
 * 3. The values in "ind" are in [1,length(G)].
 *
 * 4. The third element of each column should be positive. The program does
 *    not check the validity of the parameter. 
 *
 * 5. The values in G should be within [1, n].
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


void general_altra_yj(double *x, double *v, int n, double *G, double *ind, int nodes, double *lambda_vector){
    
    int i, j, m;
    double lambda,twoNorm, ratio;
    
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
            if (v[j]>lambda_vector[j])
                x[j]=v[j]-lambda_vector[j];
            else
                if (v[j]<-lambda_vector[j])
                    x[j]=v[j]+lambda_vector[j];
                else
                    x[j]=0;
        }
        
        i=1;
    }
    else{
        memcpy(x, v, sizeof(double) * n);
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
			twoNorm += x[(int) G[j]-1 ] * x[(int) G[j]-1 ];        
        twoNorm=sqrt(twoNorm);
        
        lambda=ind[3*i+2];
        if (twoNorm>lambda){
            ratio=(twoNorm-lambda)/twoNorm;
            
            /*
             * shrinkage this group by ratio
             */
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[(int) G[j]-1 ]*=ratio;            
        }
        else{
            /*
             * threshold this group to zero
             */
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[(int) G[j]-1 ]=0;
        }
	}
}



/*
 * altra_mt is a generalization of altra to the 
 * 
 * multi-task learning scenario (or equivalently the multi-class case)
 *
 * altra_mt(X, V, n, k, G, ind, nodes);
 *
 * It applies altra for each row (1xk) of X and V
 *
 */


void general_altra_mt_yj(double *X, double *V, int n, int k, double *G, double *ind, int nodes, double *Lambda_vector){
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
            
        general_altra_yj(x, v, k, G, ind, nodes, lambda_vector);
        
        /*
         * copy the solution to X         
         */        
        for(j=0;j<k;j++)
            X[j*n+i]=x[j];
    }
    
    free(x);
    free(v);
}




/*
 * compute
 *  lambda2_max=general_computeLambda2Max(x,n,G, ind,nodes);
 *
 * compute the 2 norm of each group, which is divided by the ind(3,:),
 * then the maximum value is returned
 */

    /*
     *This function does not consider the case ind={[-1, -1, 100]',...}
     *
     *This functions is not used currently.
     */



/*
 * -------------------------------------------------------------------
 *                       Function and parameter
 * -------------------------------------------------------------------
 *
 * treeNorm compute
 *
 *        \sum \lambda_i \|x_{G_i}\|,
 *
 * where x is of dimension n,
 *       \lambda_i >=0, and G_i's follow the tree structure
 *
 * The file is implemented in the following in Matlab:
 *
 * tree_norm=general_treeNorm(x, n, G, ind,nodes);
 */


void general_treeNorm_yj(double *tree_norm, double *x, int n, double *G, double *ind, int nodes, double *lambda_vector){
    
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
        
       // *tree_norm=*tree_norm * lambda;
        
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
			twoNorm += x[(int) G[j]-1 ] * x[(int) G[j]-1 ];        
        twoNorm=sqrt(twoNorm);
        
        lambda=ind[3*i+2];
        
        *tree_norm=*tree_norm + lambda*twoNorm;
	}

}




