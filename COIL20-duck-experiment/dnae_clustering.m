%--------------------------------------------------------------------------
% Copyright @ Dongdong, Chen 2013
%--------------------------------------------------------------------------

function [Y,eval,grp,missrate] = dnae_clustering(W,gtruth)

if (nargin < 2)
    gtruth = ones(1,size(W,1));
end

MAXiter = 1000;
REPlic = 100;
N = size(W,1);
n = max(gtruth);

% cluster the data using the normalized symmetric Laplacian 
D = diag( 1./sqrt(sum(W,1)+eps) );
L = eye(N) - D * W * D;
[U,S,V] = svd(L,'econ');
Y = V(:,end-n:end);
for i = 1:N
    Yn(i,:) = Y(i,:) ./ norm(Y(i,:)+eps);
end
eval = diag(S(end-n:end,end-n:end));

if n > 1
    grp = kmeans(Yn(:,end-n+1:end),n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
else 
    grp = ones(1,N);
end
Y = Y';

% compute the misclassification rate
missrate = missclassGroups(grp,gtruth,n) ./ length(gtruth); 
