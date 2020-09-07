function nz_mde = get_nz_mde(mde)
[m, n] = size(mde);
nz_mde = zeros(m,n);
nz_mde = mde;
[~,ids] = find(nz_mde==0);
nz_mde(ids) = [];
    