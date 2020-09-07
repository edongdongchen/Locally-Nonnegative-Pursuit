%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns a NxN coefficient matrix of the sparse representation 
% of each data point in terms of the rest of the points
% Y: DxN data matrix
% affine: true if enforcing the affine constraint, false otherwise
% thr1: stopping threshold for the coefficient error ||Z-C||
% thr2: stopping threshold for the linear system error ||Y-YZ||
% maxIter: maximum number of iterations of ALM
% C2: NxN sparse coefficient matrix
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function C2 = alm_vec_func(Y,q,lambda,mu,thr1,thr2,maxIter)

if (nargin < 3)
    % default subspaces are linear
    lambda = 10; 
end
if (nargin < 4)
    % default regularizarion parameters
    mu = 10;
end
if (nargin < 5)
    % default coefficient error threshold to stop ALM
    thr1 = 2*10^-8; 
end
if (nargin < 6)
    % default linear system error threshold to stop ALM
    thr2 = 2*10^-8; 
end
if (nargin < 7)
    % default maximum number of iterations of ALM
    maxIter = 500; 
end
[D,N] = size(Y);

A = inv((Y'*Y)+mu*eye(N)+mu*ones(N,N));
C1 = zeros(N,1);
Lambda = zeros(N,1);
gamma = 0;
err1 = 10*thr1; err2 = 10*thr2;
i = 1;
% ALM iterations
while ( (err1(i) > thr1 || err2(i) > thr2) && i < maxIter )
    % updating Z
    Z = A * (mu*C1-Lambda+gamma*ones(N,1));
    % updating C
    C2 = max(0,(abs(mu*Z+Lambda) - lambda.*q)) .* sign(mu*Z+Lambda);
    C2 = 1/mu * C2;
    % updating Lagrange multipliers
    Lambda = Lambda + mu * (Z - C2);
    gamma = gamma + mu * (1 - ones(1,N)*Z);
    % computing errors
    err1(i+1) = errorCoef(Z,C2);
    err2(i+1) = errorCoef(ones(1,N)*Z,ones(1,N));
    %
    %mu = min(mu*(1+10^-5),10^5);
    %mu2(j) = min(mu2(j)*(1+10^-4),10^5);
    %
    C1 = C2;
    i = i + 1;
end
