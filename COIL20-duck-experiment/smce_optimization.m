%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [W,C] = smce_optimization(X,lambda,K)

if (nargin < 2)
    lambda = 10;
end
if (nargin < 3)
    K = size(X,2) - 1;
end

[D,N] = size(X);
X2 = sum(X.^2,1);
Dist = sqrt( repmat(X2,N,1)+repmat(X2',1,N)-2*(X'*X) );

C = zeros(N,N); % coefficient matrix used for clustering
W = zeros(N,N); % coefficient matrix used for dimensionality reduction

% solving the SMCE optimization program
tic
for i = 1:N 
    [ds,ids] = sort(Dist(:,i),'ascend');
    ids = ids(1:K);
    
    y = X(:,ids(1));
    Y = X(:,ids);
    Y(:,1) = [];
    Y = Y - repmat(y,1,K-1);
    v = Dist(ids,i);
    v(1) = [];
    for j = 1:K-1
        Y(:,j) = Y(:,j) ./ v(j);
    end
    
%     cvx_begin
%     cvx_quiet(true)
%     variable c(K-1,1);
%     minimize lambda * norm(v.*c/sum(v),1) +  0.5 * (c' * Y') * (Y * c);
%     subject to
%     sum(c) == 1;
%     cvx_end

    c = alm_vec_func(Y,v./sum(v),lambda);

    C(ids(2:K),i) = c;
    W(ids(2:K),i) = abs(c./v) / sum(abs(c./v));
end
toc
