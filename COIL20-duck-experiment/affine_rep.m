function [repAopt,repKnn,scr] = affine_rep(A_opt, b, options)
% Affine representation: Learning 3 types of affine representation of sample
% input:
% A_opt: m x n data matrix (dictionary)
% b    : m x 1 data point    (sample)
% options: 
%        options.k      : neighborhood size
%        options.cols   : length of SCR
%        options.ids_opt: ids of A_opt in knn
%        options.ids_knn: ids of knn   in A
% output:
% repAopt: affine representation b over A_opt
% repKnn : affine representation b over knn, fill repAopt according ids_opt
% scr    : affine represnetation b over A,   fill repAopt according ids_knn 
%--------------------------------------------------------------------------
% AAAI-14 paper: A Local Non-negative Pursuit Method 
%                for Intrinsic Manifold Structure Preservation 
%--------------------------------------------------------------------------
%   version 2.0 --Sep/2014 
%   version 1.0 --Oct/2013 
%
%   Written by Dongdong Chen
%   Email: dongdongchen.scu@gmail.com

[m,~] = size(A_opt);
if (~exist('options','var'))
   options = [];
end

if ~isfield(options,'cols')
    options.cols = 10;
end

if ~isfield(options,'k') 
    options.k = 5;
end

if ~isfield(options,'ids_opt') 
    options.ids_opt = 5;
end

if ~isfield(options,'ids_knn') 
    options.ids_knn = 5;
end

ids_opt = options.ids_opt; % ids of A_opt in knn
ids_knn = options.ids_knn; % ids of knn   in SCR
n = options.cols;
n_opt = length(ids_opt);
n_knn = length(ids_knn);

repAopt = zeros(n_opt,1);
repKnn  = zeros(n_knn,1); 
scr     = zeros(n,1);

if(n>=m)
    tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
    tol=0;
end  

Z = repmat(b,1,n_opt)-A_opt;              % shift ith pt to origin
C = Z'*Z;                                 % local covariance
C = C + eye(n_opt,n_opt)*tol*trace(C);    % regularlization (K>D)
repAopt = C\ones(n_opt,1);                % solve Cx=1
repAopt = repAopt/sum(repAopt);           % enforce sum(w)=2 coding over Aopt, full rep

repKnn(ids_opt) = repAopt;
scr(ids_knn)    = repKnn;
