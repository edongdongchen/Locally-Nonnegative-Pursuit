%X1: W_smce1, Y: embedding,
function nalac = get_nalac_smce(X1, Y, isHard, alpha, lambda, Kmax)
if size(X1, 2)~= size(Y, 2)
    return;
end
[~,n] = size(X1);
[~, X2]=smce_optimization(Y, lambda, Kmax);
nalac_singles = zeros(1, n);
for i = 1:n
    x_ai = abs(X1(:,i));
    x_yi = abs(X2(:,i));
    [~, id1] = sort(x_ai, 'descend');
    [~, id2] = sort(x_yi, 'descend');
    d = min(get_intrinsicDimEstimt(x_ai, alpha), get_intrinsicDimEstimt(x_yi, alpha));
    nalac_singles(i) = get_nalac_single(id1, id2, d, isHard);
end
nalac = mean(nalac_singles);

    

function nalac_single = get_nalac_single(id1, id2, topK, isHard)
if size(id1)~= size(id2)
    return;
end
match = 0;
if isHard
    nalac_single = sum(id1(1:topK)==id2(1:topK));
    nalac_single = nalac_single/topK;
else
    for i=1:topK
        obj = id1(i);
        [v,~] = find(id2(1:topK) == obj);
        if ~isempty(v)
            match = match + 1;
        end
    end
    nalac_single = match/topK;
end