function [Yg,X,indg, missrate] = lnp_embedding(A, d, options)
if nargin < 1
     error('Too few input arguments'); 
elseif nargin < 2
     options = struct('Kmax',10,'display',1,'epsilon',0.001); 
end

if ~isfield(options,'Kmax')
     options.Kmax = 10; 
end
if ~isfield(options,'display')
     options.display = 1; 
end
if ~isfield(options,'epsilon')
     options.epsilon = 0.001; 
end
if ~isfield(options,'gtruth')
     options.gtruth = ones(1,size(A,2));
end

fprintf(1,'Spectral embdding running on %d points in %d dimensions\n', size(A,2), size(A,1));
fprintf(1,'-->LNP started.\n');
%[X, W] = lnp(A, options);
[X, W] = lnp(A, options.Kmax)
Wsym = max(W,W');
% perform clustering
fprintf(1,'-->Performing spectral clustering.\n');
[Yc,evalc,grp,missrate] = dnae_clustering(Wsym,options.gtruth);
% perform embedding
fprintf(1,'-->Performing spectral embedding.\n');
[Yg,evalg,indg] = dnae_embedding(Wsym,grp,d);

fprintf(1,'LNP Done.\n');
