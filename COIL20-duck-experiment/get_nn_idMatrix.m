function NN_idMatrix = get_nn_idMatrix(DATA)
[~,N] = size(DATA);

DATA2 = sum(DATA.^2,1);
distance = repmat(DATA2,N,1)+repmat(DATA2',1,N)-2*DATA'*DATA;

[~,index] = sort(distance);
NN_idMatrix = index(2:end,:);