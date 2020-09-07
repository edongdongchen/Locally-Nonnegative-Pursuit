function mde = get_mde(X)
X = sort(X,'descend');
mde = sum(X,2)/size(X,2);
    