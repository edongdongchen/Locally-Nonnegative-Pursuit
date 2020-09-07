function [A_opt, id_opt] = lnp_once(A_knn, b)
%lnp_once Locally Non-Negative Pursuit (LNP) for one sample.
% This function find optimal neighborhood from a linear patch (A_knn) for b.
% input:
% A_knn: DxK, K nearest neighbors (linear patch)
% b    : Dx1, sample
% output:
% A_opt : optimal neighborhood points(convex patch)
% id_opt: indices of A_opt over A_knn
%--------------------------------------------------------------------------
% AAAI-14 paper: A Local Non-negative Pursuit Method 
%                for Intrinsic Manifold Structure Preservation 
%--------------------------------------------------------------------------
%   version 2.0 --Sep/2014 
%   version 1.0 --Oct/2013 
%
%   Written by Dongdong Chen
%   Email: dongdongchen.scu@gmail.com

[m, n] = size(A_knn);
if (size(b,1)~=m)
    disp('Dimensions error');
    return
end

%% Distance initialization
A_opt = zeros(m, n);
id_opt   = zeros(1, n); 
G = repmat(b,1,n)- A_knn;
distance = sqrt(sum(G.^2,1))';

%% Neighborhood selection
k = 1;
while true 
    if k==1
        [~,id] = min(distance);
        A_opt(:,k) = A_knn(:,id);
        id_opt(k) = id;
    else
        G_k = repmat(b,1,k-1) - A_opt(:,1:k-1);% G_k
        if(k>=m)
            tol=1e-3;
        else
            tol=0;
        end  
        C = G_k'*G_k + eye(k-1,k-1)*tol*trace(G_k'*G_k);% regularlization (K>D)
        Proj_sign = sign(inv(C)*G_k'*(G));
        cands1 = (sum(Proj_sign, 1) == (1-k));          %inv(c)*c'*(d))<.0       
        cands2 = diag(distance)*cands1';
        ido = find(cands2>0); 
        if size(ido,1)~=0
            [~,t] = min(distance(ido));                             
            A_opt(:,k) = A_knn(:,ido(t));
            id_opt(k) = ido(t);   
        else
            break;                                                       
        end
    end
    k = k+1;    
end
k = k-1;

%% Return
A_opt    = A_opt(:,1:k);
id_opt   = id_opt(1:k); 