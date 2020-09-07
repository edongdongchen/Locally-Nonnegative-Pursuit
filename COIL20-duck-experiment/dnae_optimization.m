function [X,W] = dnae_optimization(A,options)

if nargin < 1
     error('Too few input arguments'); 
elseif nargin < 2
     options = struct('Kmax',10,'display',1,'epsilon',0.001); 
end

[X, W] = lnp(A, options);