function  [Lx,bx]=iter_xFotv(g,bx,w,mu,a)

[~,n,~] = size(g);
% forward finite difference operator
gx = imfilter(g,w,'circular','full'); % column differences
gx = gx(:,1:n,:);

% update dx
dx = shrink_operator(gx,bx,mu);

% update bx
bx = bx + (gx - dx);

% backward difference 
% Transpose of the backward finite difference operator
d = length(w);
w = fliplr(w);
Lx = imfilter(dx-bx,w,'circular','full');
Lx = Lx(:,d:end,:);
Lx = conj((-1)^a)*(-1)^a*Lx;







