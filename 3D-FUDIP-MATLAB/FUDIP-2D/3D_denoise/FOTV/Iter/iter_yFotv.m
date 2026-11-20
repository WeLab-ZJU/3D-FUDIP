function  [Ly,by]=iter_yFotv(g,by,w,mu,a)

[m,~,~] = size(g);
% forward finite difference operator
gy = imfilter(g,w,'circular','full'); % column differences
gy = gy(1:m,:,:);

% update dx
dy = shrink_operator(gy,by,mu);

% update bx
by = by + (gy - dy);

% backward difference 
% Transpose of the backward finite difference operator
d = length(w);
w = fliplr(w);
Ly = imfilter(dy-by,w,'circular','full');
Ly = Ly(d:end,:,:);
Ly = conj((-1)^a)*(-1)^a*Ly;
