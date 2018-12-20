function [x, funVal, ValueL]=mt_sgLeastR_yj(A, y, z, opts, CTLTR)
%
%%
% my Function sgLeastR
%      Least Squares Loss with the 
%           sparse group Lasso Regularization
%
%% Problem
%
%  min  1/2 || A x - y||^2 + z_1 \|x\|_1 + z_2 * sum_j w_j ||x_{G_j}||
%
%  G_j's are nodes with tree structure
%
%  For this special case, 
%    we have L1 for each element
%    and the L2 for the non-overlapping group 
%
%  The tree overlapping group information is contained in
%  opts.ind, which is a 3 x nodes matrix, where nodes denotes the number of
%  nodes of the tree.
%  opts.ind(1,:) contains the starting index
%  opts.ind(2,:) contains the ending index
%  opts.ind(3,:) contains the corresponding weight (w_j)
%
%
%
%% Input parameters:
%
%  A-         Matrix of size m x n
%                A can be a dense matrix
%                         a sparse matrix
%                         or a DCT matrix
%  y -        Response vector (of size mx1)
%  z -        The regularization parameter (z=[z_1,z_2] >=0)
%  opts-      Optional inputs (default value: opts=[])
%
%% Output parameters:
%
%  x-         Solution
%  funVal-    Function value during iterations
%
%% Copyright (C) 2010-2011 Jun Liu, and Jieping Ye
%
% You are suggested to first read the Manual.
%
% For any problem, please contact with Jun Liu via j.liu@asu.edu
%
% Last modified on April 21, 2010.
%
%% Related papers
%
% [1] Jun Liu and Jieping Ye, Moreau-Yosida Regularization for 
%     Grouped Tree Structure Learning, NIPS 2010
%
%% Related functions:
%
%  sll_opts
%
%%

%% Verify and initialize the parameters
%%
if (nargin <4)
    error('\n Inputs: A, y, z, and opts (.ind) should be specified!\n');
end

[m,n]=size(A);

if (length(y) ~=m)
    error('\n Check the length of y!\n');
end

if (z(1)<0 || z(2)<0)
    error('\n z should be nonnegative!\n');
end

lambda1=z(1)*CTLTR;
lambda2=z(2);

opts=sll_opts(opts); % run sll_opts to set default values (flags)

% restart the program for better efficiency
%  this is a newly added function
if (~isfield(opts,'rStartNum'))
    opts.rStartNum=opts.maxIter;
else
    if (opts.rStartNum<=0)
        opts.rStartNum=opts.maxIter;
    end
end

%% Detailed initialization
%% Normalization

% Please refer to sll_opts for the definitions of mu, nu and nFlag
%
% If .nFlag =1, the input matrix A is normalized to
%                     A= ( A- repmat(mu, m,1) ) * diag(nu)^{-1}
%
% If .nFlag =2, the input matrix A is normalized to
%                     A= diag(nu)^{-1} * ( A- repmat(mu, m,1) )
%
% Such normalization is done implicitly
%     This implicit normalization is suggested for the sparse matrix
%                                    but not for the dense matrix
%

if (opts.nFlag~=0)
    if (isfield(opts,'mu'))
        mu=opts.mu;
        if(size(mu,2)~=n)
            error('\n Check the input .mu');
        end
    else
        mu=mean(A,1);
    end
    
    if (opts.nFlag==1)
        if (isfield(opts,'nu'))
            nu=opts.nu;
            if(size(nu,1)~=n)
                error('\n Check the input .nu!');
            end
        else
            nu=(sum(A.^2,1)/m).^(0.5); nu=nu';
        end
    else % .nFlag=2
        if (isfield(opts,'nu'))
            nu=opts.nu;
            if(size(nu,1)~=m)
                error('\n Check the input .nu!');
            end
        else
            nu=(sum(A.^2,2)/n).^(0.5);
        end
    end
    
    ind_zero=find(abs(nu)<= 1e-10);    nu(ind_zero)=1;
    % If some values in nu is typically small, it might be that,
    % the entries in a given row or column in A are all close to zero.
    % For numerical stability, we set the corresponding value to 1.
end

if (~issparse(A)) && (opts.nFlag~=0)
    fprintf('\n -----------------------------------------------------');
    fprintf('\n The data is not sparse or not stored in sparse format');
    fprintf('\n The code still works.');
    fprintf('\n But we suggest you to normalize the data directly,');
    fprintf('\n for achieving better efficiency.');
    fprintf('\n -----------------------------------------------------');
end

%% Group & Others 

% Initialize ind
if ~isfield(opts,'ind')
    error('\n In tree_mtLeastR, .ind should be specified');
else
    ind=opts.ind;
    k=length(ind)-1;
    
    if ind(k+1)~=m
        error('\n Check opts.ind');
    end
end

% Initialize ind 
if (~isfield(opts,'ind_t'))
    error('\n In tree_mtLeastR, the field .ind_t should be specified');
else
    ind_t=opts.ind_t;
   
    if (size(ind_t,1)~=3)
        error('\n Check opts.ind_t');
    end
end

GFlag=0;
% if GFlag=1, we shall apply general_altra
if (isfield(opts,'G'))
    GFlag=1;
    G=opts.G;
end


%% Starting point initialization

ATy=zeros(n, k);
% compute AT y
for i=1:k
    ind_i=(ind(i)+1):ind(i+1);
    
    if (opts.nFlag==0)
        tt =A(ind_i,:)'*y(ind_i,1);
    elseif (opts.nFlag==1)
        tt= A(ind_i,:)'*y(ind_i,1) - sum(y(ind_i,1)) * mu';  
        tt=tt./nu(ind_i,1);
    else
        invNu=y(ind_i,1)./nu(ind_i,1);
        tt=A(ind_i,:)'*invNu - sum(invNu)*mu';
    end
    
    ATy(:,i)= tt;
end

% process the regularization parameter
if (opts.rFlag==0)
    lambda=z;
else % z here is the scaling factor lying in [0,1]
%     if (lambda1<0 || lambda1>1 || lambda2<0 || lambda2>1)
%         error('\n opts.rFlag=1, and z should be in [0,1]');
%     end
    
    % compute lambda1_max
    temp=abs(ATy);

    lambda1_max=max(max(temp));
    
    lambda1=lambda1*lambda1_max;
    
    % compute lambda2_max(lambda_1)
    temp=max(temp-lambda1,0);
        
    if ( min(ind_t(3,:))<=0 )
        error('\n In this case, lambda2_max = inf!');
    end
    
    %lambda2_max=computeLambda2Max(temp,n,ind_t,size(ind_t,2));
    lambda2_max=sqrt( max( sum(temp.^2,2) ) );
    lambda2=lambda2*lambda2_max;
    %% 这里不太懂
%     computedFlag=0;
%     if (isfield(opts,'lambdaMax'))
%         if (opts.lambdaMax~=-1)
%             lambda=z*opts.lambdaMax;
%             computedFlag=1;
%         end
%     end
%     
%     if (~computedFlag)
%         if (GFlag==0)
%             lambda_max=findLambdaMax_mt(ATy, n, k, ind_t, size(ind_t,2));%如果没有定义G，这还是需要去改那个C函数啊
%         else
%             lambda_max=general_findLambdaMax_mt(ATy, n, k, G, ind_t, size(ind_t,2)); %如果定义了G
%         end
%         
%         % As .rFlag=1, we set lambda as a ratio of lambda_max
%         lambda=z*lambda_max;
%     end
% 而且也没有定义下面的opts.lambdaMax
end

% initialize a starting point
if opts.init==2
    x=zeros(n,k);
else
    if isfield(opts,'x0')
        x=opts.x0;
        if (length(x)~=n)
            error('\n Check the input .x0');
        end
    else
        x=ATy;  % if .x0 is not specified, we use ratio*ATy,
        % where ratio is a positive value
    end
end

Ax=zeros(m,1);
% compute Ax: Ax_i= A_i * x_i
for i=1:k    
    ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
    m_i=ind(i+1)-ind(i);          % number of samples in the i-th group
    
    if (opts.nFlag==0)
        Ax(ind_i,1)=A(ind_i,:)* x(:,i);
    elseif (opts.nFlag==1)
        invNu=x(:,i)./nu; mu_invNu=mu * invNu;
        Ax(ind_i,1)=A(ind_i,:)*invNu -repmat(mu_invNu, m_i, 1);
    else
        Ax(ind_i,1)=A(ind_i,:)*x(:,i)-repmat(mu*x(:,i), m, 1);    
        Ax(ind_i,1)=Ax./nu(ind_i,1);
    end
end

if (opts.init==0) 
    % ------  This function is not available
    %
    % If .init=0, we set x=ratio*x by "initFactor"
    % Please refer to the function initFactor for detail
    %
    
    % Here, we only support starting from zero, due to the complex tree
    % structure
    
    x=zeros(n,k);    
end

%% The main program
% The Armijo Goldstein line search schemes + accelearted gradient descent

if (opts.mFlag==0 && opts.lFlag==0)  
    
    bFlag=0; % this flag tests whether the gradient step only changes a little

    L=1;
    % We assume that the maximum eigenvalue of A'A is over 1
    
    % assign xp with x, and Axp with Ax
    xp=x; Axp=Ax; xxp=zeros(n,k);
    
    alphap=0; alpha=1;
    
    for iterStep=1:opts.maxIter
        % --------------------------- step 1 ---------------------------
        % compute search point s based on xp and x (with beta)
        beta=(alphap-1)/alpha;    s=x + beta* xxp;
        
        % --------------------------- step 2 ---------------------------
        % line search for L and compute the new approximate solution x
        
        % compute the gradient (g) at s
        As=Ax + beta* (Ax-Axp);
        
        % compute ATAs : n x k
        for i=1:k
            ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
            
            if (opts.nFlag==0)
                tt =A(ind_i,:)'*As(ind_i,1);
            elseif (opts.nFlag==1)
                tt= A(ind_i,:)'*As(ind_i,1) - sum(As(ind_i,1)) * mu';
                tt=tt./nu(ind_i,1);
            else
                invNu=As(ind_i,1)./nu(ind_i,1);
                tt=A(ind_i,:)'*invNu - sum(invNu)*mu';
            end
            
            ATAs(:,i)= tt;
        end
        
        % obtain the gradient g
        g=ATAs-ATy;
        
        % copy x and Ax to xp and Axp
        xp=x;    Axp=Ax;
        
        while (1)
            % let s walk in a step in the antigradient of s to get v
            % and then do the L1/Lq-norm regularized projection
            v=s-g/L;
            
            % tree overlapping group Lasso projection
            ind_work(1:2,:)=[ [-1, -1]', ind_t(1:2,:) ];
            ind_work(3,:)=[ z(1)*lambda1_max/L, ind_t(3,:) * (lambda2 / L) ];
           % ind_work(:,2)=[];
            lambda_vector=lambda1/L;
            if (GFlag==0)
            %x=altra_yj(v, n, ind_work, size(ind_work,2),lambda_vector);
                for  ii=1:k
                    vv=v(:,ii); lv=lambda_vector(:,ii); 
                    xx=altra_yj(vv, n, ind_work, size(ind_work,2),lv);
                    x(:,ii)=xx;
                end
                % x=altra_mt_yj(v, n, k, ind_work, size(ind_work,2),lambda_vector);
            else
                for  ii=1:k
                    vv=v(:,ii); lv=lambda_vector(:,ii); GG=G(n*(ii-1)+1:n*ii); 
                    indd=cat(2,ind_work(:,1),ind_work(:,2*ii),ind_work(:,2*ii+1));
                    xx=general_altra_yj(vv, n, GG, indd, size(indd,2),lv);
                    x(:,ii)=xx;
                end
               % x=general_altra_mt_yj(v, n, k, G, ind_work, size(ind_work,2),lambda_vector);
            end
            
            v=x-s;  % the difference between the new approximate solution x
            % and the search point s
            
            % compute A x
            
            for i=1:k
                ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
                m_i=ind(i+1)-ind(i);          % number of samples in the i-th group
                
                if (opts.nFlag==0)
                    Ax(ind_i,1)=A(ind_i,:)* x(:,i);
                elseif (opts.nFlag==1)
                    invNu=x(:,i)./nu; mu_invNu=mu * invNu;
                    Ax(ind_i,1)=A(ind_i,:)*invNu -repmat(mu_invNu, m_i, 1);
                else
                    Ax(ind_i,1)=A(ind_i,:)*x(:,i)-repmat(mu*x(:,i), m, 1);
                    Ax(ind_i,1)=Ax./nu(ind_i,1);
                end
            end
            
            Av=Ax -As;
            r_sum=norm(v,'fro')^2;  l_sum=Av'*Av;
            
            if (r_sum <=1e-20)
                bFlag=1; % this shows that, the gradient step makes little improvement
                break;
            end
            
            % the condition is ||Av||_2^2 <= L * ||v||_2^2
            if(l_sum <= r_sum * L)
                break;
            else
                L=max(2*L, l_sum/r_sum);
                %fprintf('\n L=%5.6f',L);
            end
        end
        
        % --------------------------- step 3 ---------------------------
        % update alpha and alphap, and check whether converge
        alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;
        
        xxp=x-xp;   Axy=Ax-y;
        
        ValueL(iterStep)=L;
        
        % compute the regularization part
        ind_work(1:2,:)=[ [-1, -1]', ind_t(1:2,:) ];
        ind_work(3,:)=[ z(1)*lambda1_max, ind_t(3,:) * lambda2 ];
       % ind_work(:,2)=[];

        % tree_norm=treeNorm_yj(x, n, ind_work, size(ind_work,2), lambda_vector);
        % tree_norm=treeNorm(x, n, ind_work, size(ind_work,2));
        % function value = loss + regularizatioin
        funVal(iterStep)=Axy'* Axy/2;
        for i=1:k
            xRow=x(:,i);            
            lambda_vector_Row=lambda_vector(:,i);
            GGG=G(n*(ii-1)+1:n*ii); 
            inddd=cat(2,ind_work(:,1),ind_work(:,2*i),ind_work(:,2*i+1));
            if (GFlag==0)
                tree_norm=treeNorm_yj(xRow, k, inddd, size(inddd,2), lambda_vector_Row);
            else
                tree_norm=general_treeNorm_yj(xRow, k, GGG, inddd, size(inddd,2), lambda_vector_Row);
            end
            
            funVal(iterStep)=funVal(iterStep)+ tree_norm;
        end 
        
        if (bFlag)
            % fprintf('\n The program terminates as the gradient step changes the solution very small.');
            break;
        end
        
        switch(opts.tFlag)
            case 0
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <= opts.tol)
                        break;
                    end
                end
            case 1
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <=...
                            opts.tol* funVal(iterStep-1))
                        break;
                    end
                end
            case 2
                if ( funVal(iterStep)<= opts.tol)
                    break;
                end
            case 3
                norm_xxp=sqrt(xxp'*xxp);
                if ( norm_xxp <=opts.tol)
                    break;
                end
            case 4
                norm_xp=sqrt(xp'*xp);    norm_xxp=sqrt(xxp'*xxp);
                if ( norm_xxp <=opts.tol * max(norm_xp,1))
                    break;
                end
            case 5
                if iterStep>=opts.maxIter
                    break;
                end
        end
        
        % restart the program every opts.rStartNum
        if (~mod(iterStep, opts.rStartNum))
            alphap=0; alpha=1;
            xp=x; Axp=Ax; xxp=zeros(n,1); L =L/2;
        end
    end
else
    error('\n The function does not support opts.mFlag neq 0 & opts.lFlag neq 0!');
end