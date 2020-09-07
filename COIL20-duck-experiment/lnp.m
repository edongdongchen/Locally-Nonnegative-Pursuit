function [SCRs, W] = lnp(A, K)
% lnp LOCALLY NON-NEGATIVE PURSUIT.
% 1. Finding a sparse and convex neighborhood
% 2. Learning sparse convex representations (SCRs) of A
% 3. Performing local structure preserving embedding (spectral clustering)
% input:
% A      -   DxN data matrix (dictionary)
% K      -   1x1 max neighborhood size 
%
% output:
% SCRs   -   optimal sparse convex representaitons. The input of spectral
%            clustering algo.
%--------------------------------------------------------------------------
% AAAI-14 paper: A Local Non-negative Pursuit Method 
%                for Intrinsic Manifold Structure Preservation 
%--------------------------------------------------------------------------
%   version 2.0 --Sep/2014
%   version 1.0 --Oct/2013 
%
%   Written by Dongdong Chen
%   Email: dongdongchen.scu@gmail.com

if nargin < 1
     error('Input arguments illlegal'); 
end
[D,N] = size(A);
if K >= N
     error('K must less than number of samples'); 
end
if (~exist('K','var'))
   K = 5;
end

options = [];
SCRs    = zeros(N,N);
W       = zeros(N,N);

A2             = sum(A.^2,1);
distance       = repmat(A2,N,1)+repmat(A2',1,N)-2*A'*A;
[~,index]      = sort(distance);
neighborhood   = index(2:(1 + K),:);

for i=1:N
    ids = neighborhood(:,i);
    knn = A(:,ids);
    b   = A(:,i);

    [A_opt, ids_opt] = lnp_once(knn, b);               %  LNP ---> select neighbors

    options.k       = K;
    options.ids_opt = ids_opt;                         % indexs of A_opt in knn
    options.ids_knn = neighborhood(:,i);               % indexs of knn   in SCR
    options.cols    = N;

    ids_scr =  neighborhood(ids_opt,i);
    
    [repAopt,~,scr] = affine_rep(A_opt, b, options);   % SCR ---> optimal representation

    SCRs(:,i) = scr;                                   % Assignment
    W(ids_scr,i) = (repAopt./distance(ids_scr,i))/sum(repAopt./distance(ids_scr,i));
end