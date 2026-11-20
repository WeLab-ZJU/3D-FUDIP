function  [Lz,bz]=iter_zFotv(g,bz,w,mu,a)

[~,~,t] = size(g);
% forward finite difference operator
gz = imfilter(g,w,'circular','full'); % column differences
gz = gz(:,:,1:t);

% update dx
dz = shrink_operator(gz,bz,mu);

% update bx
bz = bz + (gz - dz);

% backward difference 
% Transpose of the backward finite difference operator
d = length(w);
w = fliplr(w);
Lz = imfilter(dz-bz,w,'circular','full');
Lz = Lz(:,:,d:end);
Lz = conj((-1)^a)*(-1)^a*Lz;