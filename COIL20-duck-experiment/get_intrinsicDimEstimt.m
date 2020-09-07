function dim = get_intrinsicDimEstimt(X, alpha);
mde = get_mde(X);
[D,~] = size(mde);
for i=1:D
    t1 = sum(mde(1:i));
    t2 =sum(mde);
     if t1>= alpha*t2;
         dim= i;
         break;
     end
end
dim = dim -1;

