%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Y,Eval] = SpectralEmbedding(W,d)

N = size(W,1);

D = diag( 1./sqrt(sum(W,1)+eps) );
L = eye(N) - D * W * D;
[U,S,V] = svd(L);
Y = D * V(:,end-d+1:end);
Eval = diag(S(end-d+1:end,end-d+1:end));