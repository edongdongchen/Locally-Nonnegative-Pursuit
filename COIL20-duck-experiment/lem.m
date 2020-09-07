% --- leigs function for Laplacian eigenmap.
% Written by Belkin & Niyogi, 2002.[E,V] = leigs(handles.X, 'nn', handles.K, handles.d+1);
function [Y,Wl] = lem(DATA, K, d) 
DATA = DATA';
n = size(DATA,1);
A = sparse(n,n);
step = 10;
fprintf(1,'LEM running on %d points in %d dimensions\n',size(DATA,2),size(DATA,1));
for i1=1:step:n    
    i2 = i1+step-1;
    if (i2> n) 
      i2=n;
    end;
    XX= DATA(i1:i2,:);  
    dt = L2_distance(XX',DATA',0);
    [Z,I] = sort ( dt,2);
    for i=i1:i2
      for j=2:K+1
	        A(i,I(i-i1+1,j))= Z(i-i1+1,j); 
	        A(I(i-i1+1,j),i)= Z(i-i1+1,j); 
      end;    
    end;
end;
W = A;
[A_i, A_j, A_v] = find(A);  % disassemble the sparse matrix
Wl = zeros(n,n);
for i = 1: size(A_i)  
    W(A_i(i), A_j(i)) = 1;
    Wl(A_i(i), A_j(i)) = 1;
end;
D = sum(W(:,:),2);   
L = spdiags(D,0,speye(size(W,1)))-W;
opts.tol = 1e-9;
opts.issym=1; 
opts.disp = 0; 
% [Y,eigenvals] = eigs(M,d+1,0,options);
[Y,V] = eigs(L,d+1,'sm',opts);

n=size(Y,2);
Y = Y(:,n-d:n-1)'*sqrt(size(DATA,2));


fprintf(1,'LEM Done.\n');

