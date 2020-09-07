% An example of neighborhood selection of LNP
%--------------------------------------------------------------------------
% AAAI-14 paper: A Local Non-negative Pursuit Method 
%                for Intrinsic Manifold Structure Preservation 
%--------------------------------------------------------------------------
%   version 2.0 --Sep/2014 
%   version 1.0 --Oct/2013 
%
%   Written by Dongdong Chen
%   Email: dongdongchen.scu@gmail.com

clc, clear all, warning off; clf; whitebg;

D = 2;                       % dimension
N = 200;                     % # points
A = randi([-N,N],D,N);       %randomly create a dictionary
b = randi([-N,N],D,1);     	 %randomly create a observed point

K = 10;                  	 %maximal neighborhood size

distance = sqrt(sum((repmat(b,1,N) - A).^2,1));

[~, nn]  = sort(distance);
A_knn    = A(:,nn(1:K));     % KNN of b over A

tic;
[A_opt, nn_opt] = lnp_once(A_knn, b);
toc;

options.k    = K;
options.cols = N;
options.ids_knn = nn(1:K);
options.ids_opt = nn_opt;

[~,~,scr] = affine_rep(A_opt, b, options);

figure(1);
plot(A(1,:),A(2,:),'g.');
hold on;
plot(b(1),b(2),'b*');
hold on;
scatter(A_opt(D-1,:), A_opt(D,:), 30, 'ro');
hold on;
legend('points in A','b','A_{opt}','Location','Best');
title('LNP');
hold off;

figure(2);
stem(1:N,scr);
s1 = strcat('# iterations:',num2str(sum(scr~=0)));
s2 = strcat('     reconstruction error=', num2str(norm(b - A*scr,2))); 
title(strcat( s1 , s2));
xlabel('index of points in A');
ylabel('sparse convex representation: x_i');
fprintf('Reconstruction error e = %d',norm(b - A*scr,2)); 

