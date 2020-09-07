function nalac = get_nalac(idMatrix1, idMatrix2, topK, isHard)
if size(idMatrix1)~= size(idMatrix2)
    return;
end
[~,n] = size(idMatrix1);
nalac_singles = zeros(1, n);
for i = 1:n
    nalac_singles(i) = get_nalac_single(idMatrix1(:,i), idMatrix2(:,i), topK, isHard);
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