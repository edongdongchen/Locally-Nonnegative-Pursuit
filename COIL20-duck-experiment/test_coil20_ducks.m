clc;
warning off; clf; whitebg;
load('duck_32_32.mat') ;

%% Data initialization
A = duck_32_32;
gtruth = ones(1, 72);
[D,N] = size(A);
n = max(gtruth);
clear duck_32_32;
%% Data show

tt = 1:3:72;
A_show = A(:,tt);

canonicalImageSize = [ 32 32];
numImage = 24;
layout.xI = 2;
layout.yI = 12;
layout.gap = 2;
layout.gap2 = 1;
% layout
xI = layout.xI ;
yI = layout.yI ;
gap = layout.gap ;
gap2 = layout.gap2 ; 

container = ones(canonicalImageSize(1)+gap, canonicalImageSize(2)+gap); 
% white edges
bigpic = cell(xI,yI); % (xI*canonicalImageSize(1),yI*canonicalImageSize(2));

for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(canonicalImageSize(1)+gap, canonicalImageSize(2)+gap);
        else
            container ((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(A_show(:,yI*(i-1)+j), canonicalImageSize);
            bigpic{i,j} = container;
        end
    end
end
f = 1;
figure(f)
title('COIL-20 DUCK examples') ;
imshow(cell2mat(bigpic),[],'DisplayRange',[0 max(max(A_show))],'Border','tight')

%% parameter setting
k_start = 1; k_end = 6;
ui_row  = k_end - k_start+1;
ui_col  = 8;%4
ui_step = 1;

embeddings = cell(ui_row,ui_col/2);

Ks = [2,6,10,15,40,70];
Lambdas = 60./Ks;

%% recoder initialization
tocs_lnp = zeros(1,ui_row);
tocs_lle  = zeros(1,ui_row);
tocs_lem  = zeros(1,ui_row);
tocs_smce = zeros(1,ui_row);

negN_lnp = zeros(1,ui_row);
negN_lle  = zeros(1,ui_row);
negN_lem  = zeros(1,ui_row);
negN_smce = zeros(1,ui_row);

mde_lnp   = zeros(N, ui_row);  %mde: Average representation (sorted in descending order)
mde_lle   = zeros(N, ui_row);
mde_lem   = zeros(N, ui_row);
mde_smce  = zeros(N, ui_row);
%% manifold learning start
for ite = k_start:k_end
    d = 2;
   %% RUN LNP ALGORITHM
    options.Kmax = Ks(ite);                              %maximam neighborhood size
    options.display = 1;                             %show the result
    options.epsilon = 0.01;                          %convergence condition
    options.gtruth = gtruth;                         %class label
    K = options.Kmax;

    tic
    [Y, W, indg, missrate]=lnp_embedding(A,d,options);
    toc
    fprintf(1,'-->LNP is done.\n');
    
    tocs_lnp(ui_step)  = toc;
    negN_lnp(ui_step) = length(find(W<0))/(N^2);
    mde_lnp(:, ui_step) = get_mde(W);
    
    
    %% RUN LLE ALGORITHM
    tic
    [~,W_lle]=lle(A,K,d);
    W_lle = processC(W_lle,0.95);
    % symmetrize the adjacency matrices
    Wsym_lle = max(abs(W_lle),abs(W_lle)');
    % perform clustering
    fprintf(1,'-->LLE Performing spectral clustering.\n');
    [Yc,evalc,grp,missrate_lle] = dnae_clustering(Wsym_lle,options.gtruth);
    % perform embedding
    fprintf(1,'-->LLE Performing spectral embedding.\n');
    [Y_lle, evalg, indg_lle] = dnae_embedding(Wsym_lle,grp,d);
    fprintf(1,'LLE is done.\n');
    toc
    
    tocs_lle(ui_step) = toc;
    negN_lle(ui_step) = length(find(W_lle<0))/(N^2);
    mde_lle(:, ui_step) = get_mde(W_lle);
    
    %% RUN LEM ALGORITHM
    tic
    [~,W_lem]=lem(A,K,d);

    W_lem = processC(W_lem,0.95);
    % symmetrize the adjacency matrices
    Wsym_lem = max(abs(W_lem),abs(W_lem)');
    % perform clustering
    fprintf(1,'-->LEM Performing spectral clustering.\n');
    [Yc,evalc,grp,missrate_lem] = dnae_clustering(Wsym_lem,options.gtruth);
    % perform embedding
    fprintf(1,'-->LEM Performing spectral embedding.\n');
    [Y_lem, evalg, indg_lem] = dnae_embedding(Wsym_lem,grp,d);
    fprintf(1,'LEM is done.\n');
    toc

    tocs_lem(ui_step) = toc;
    negN_lem(ui_step) = length(find(W_lem<0))/(N^2);
    mde_lem(:, ui_step) = get_mde(W_lem);
    %% RUN SMCE ALGORITHM
    
    tic
    lambda = Lambdas(ite);
    [~,W_smce]=smce_optimization(A,lambda,50);
    W_smce = processC(W_smce,0.95);
    % symmetrize the adjacency matrices
    Wsym_smce = max(abs(W_smce),abs(W_smce)');
    % perform clustering
    fprintf(1,'-->SMCE Performing spectral clustering.\n');
    [Yc,evalc,grp,missrate_smce] = dnae_clustering(Wsym_smce,options.gtruth);
    % perform embedding
    fprintf(1,'-->SMCE Performing spectral embedding.\n');
    [Y_smce, evalg, indg_smce] = dnae_embedding(Wsym_smce,grp,d);
    fprintf(1,'SMCE is done.\n');
    toc
    
    tocs_smce(ui_step) = toc;
    negN_smce(ui_step) = length(find(W_smce<0))/(N^2);
    mde_smce(:, ui_step) = get_mde(W_smce);
    
    embeddings{ite,1} = Y{1};
    embeddings{ite,2} = Y_lle{1};
    embeddings{ite,3} = Y_lem{1};
    embeddings{ite,4} = Y_smce{1};
    
   % plot the embedding of LNP
    for i = 1:n %n is the number of classes
        color{i} = jet(N);
        color{i} = color{i}(indg{i},:);
        figure(f+1)
        subplot(ui_row,ui_col,(ui_step-1)*ui_col+i);
        for j = 1:size(Y{i}(end-1,:),2)
            plot(Y{i}(end-1,j),Y{i}(end,j),'o','color',color{i}(j,:),'MarkerFaceColor',color{i}(j,:),'MarkerSize',1.6)
            hold on
        end

        axis('equal');
        
        axis off;
        box off;
        
        title(strcat('K = ',num2str(options.Kmax)));
        if ite == k_end
            xlabel('LNP');
        end
    end

    figure(f+1)
    subplot(ui_row,ui_col,(ui_step-1)*ui_col+i+1);
    bar(get_nz_mde(mde_lnp(:,ui_step)'));
    title('LNP');
    box off;
    
    
    % plot the embedding of LLE
    for i = 1:n
        color{i} = jet(N);
        color{i} = color{i}(indg_lle{i},:);
        figure(f+1)
        subplot(ui_row,ui_col,(ui_step-1)*ui_col+i+2);
        for j = 1:size(Y_lle{i}(end-1,:),2)
            plot(Y_lle{i}(end-1,j),Y_lle{i}(end,j),'o','color',color{i}(j,:),'MarkerFaceColor',color{i}(j,:),'MarkerSize',1.6)
            hold on
        end
        axis('equal');
        axis off;
        box off;
        title(strcat('K = ',num2str(options.Kmax)));
        if ite == k_end
            xlabel('LLE');
        end
    end

    figure(f+1)
    subplot(ui_row,ui_col,(ui_step-1)*ui_col+i+3);
    bar(get_nz_mde(mde_lle(:,ui_step)'));
    title('LLE');
    axis normal;
    set(gca, 'XTick', [0 1 2]);
    box off;
    
    % plot the embedding of LEM
    for i = 1:n
        color{i} = jet(N);
        color{i} = color{i}(indg_lem{i},:);
        figure(f+1)
        subplot(ui_row,ui_col,(ui_step-1)*ui_col+i+4);
        for j = 1:size(Y_lem{i}(end-1,:),2)
            plot(Y_lem{i}(end-1,j),Y_lem{i}(end,j),'o','color',color{i}(j,:),'MarkerFaceColor',color{i}(j,:),'MarkerSize',1.6)
            hold on
        end
        axis('equal');
        axis off;
        box off;
        title(strcat('K = ',num2str(options.Kmax)));
        if ite == k_end
            xlabel('LEM');
        end        
    end

    figure(f+1)
    subplot(ui_row,ui_col,(ui_step-1)*ui_col+i+5);
    ttt = get_nz_mde(mde_lem(:,ui_step)');
    bar(ttt);
    title('LEM');
    box off;
    
    
    % plot the embedding of SMCE
    for i = 1:n %n is the number of classes
        color{i} = jet(N);
        color{i} = color{i}(indg_smce{i},:);
        figure(f+1)
        subplot(ui_row,ui_col,(ui_step-1)*ui_col+i+6);
        for j = 1:size(Y_smce{i}(end-1,:),2)
            plot(Y_smce{i}(end-1,j),Y_smce{i}(end,j),'o','color',color{i}(j,:),'MarkerFaceColor',color{i}(j,:),'MarkerSize',1.6)
            hold on
        end
        axis('equal');
        axis off;
        box off;
        title(strcat('\lambda = ',num2str(lambda)));   
        if ite == k_end
            xlabel('SMCE');
        end          
    end
    
    figure(f+1)
    subplot(ui_row,ui_col,(ui_step-1)*ui_col+i+7);
    bar(get_nz_mde(mde_smce(:,ui_step)'));
    title('SMCE');
    box off;

    ui_step = ui_step + 1;
    
end


%% RESULT VISUALIZATION

% Visualization of the negative components percentage
figure(f+2);
plot(Ks,negN_lle,'-g^','LineWidth',2,'MarkerSize',7);
hold on;
plot(Ks,negN_lem,'-bo','LineWidth',2,'MarkerSize',7);
hold on;
plot(Ks,negN_smce,'-md','LineWidth',2,'MarkerSize',7);
hold on;
plot(Ks,negN_lnp,'-rs','LineWidth',2,'MarkerSize',7);
hold on;
legend('LLE','LEM','SMCE','LNP','Location','Best');
xlabel('Neighborhood size(\lambda)');
ylabel('Negative components(%)');
hold off;


% Visualization of the time cost
figure(f+3);
plot(Ks,tocs_lle,'-g*','LineWidth',2,'MarkerSize',7);
hold on;
plot(Ks,tocs_lem,'-b^','LineWidth',2,'MarkerSize',7);
hold on;
plot(Ks,tocs_smce,'-ms','LineWidth',2,'MarkerSize',7);
hold on;
plot(Ks,tocs_lnp,'-ro','LineWidth',2,'MarkerSize',7);
hold on;
legend('LLE','LEM','SMCE','LNP','Location','Best');
xlabel('Neighborhood size');
ylabel('Time cost (s)');
hold off;
