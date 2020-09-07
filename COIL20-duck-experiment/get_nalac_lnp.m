function nalac = get_nalac_lnp(A, Y, isHard, alpha)
if size(A, 2)~= size(Y, 2)
    return;
end
[~,n] = size(A);
nalac_singles = zeros(1, n);
for i = 1:n
    ai = A(:,i);
    yi = Y(:,i);
    Ai = A;
    Yi = Y;
    Ai(:,i) = [];
    Yi(:,i) = [];
    options.Kmax = 100;
    [x_ai, id1] = lnp_once(Ai, ai, options);
    [x_yi, id2] = lnp_once(Yi, yi, options);
        
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
        [v,~] = find(id2(:) == obj);
        if ~isempty(v)
            match = match + 1;
        end
    end
    nalac_single = match/topK;
end