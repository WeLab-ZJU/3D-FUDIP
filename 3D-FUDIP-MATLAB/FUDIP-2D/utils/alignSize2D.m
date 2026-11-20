function img2 = alignSize2D(img1, Sx2, Sy2, padValue)
% USAGE: img2 = Align_size_2d(img1, Sx2, Sy2, padValue)
%
% Description
% -----------  
%   Align the size of a 2D image or point-spread function (PSF) to a target 
%   size (Sx2, Sy2). It performs symmetric padding (with a specified value) 
%   if the target size is larger, or cropping from the center if the target 
%   size is smaller.
%
% INPUT:
%        'img1'         -  the input 2D image or PSF
%        'Sx2'          -  the target size in the first dimension (rows)
%        'Sy2'          -  the target size in the second dimension (columns)
%        'padValue'     -  (optional) the value used for padding (default 0)
%
% OUTPUT:
%        'img2'         -  the resulting image with the target size (Sx2, Sy2)


if(nargin == 3)
    padValue = 0;
end

[Sx1, Sy1] = size(img1);
Sx = max(Sx1, Sx2);
Sy = max(Sy1, Sy2);

imgTemp = ones(Sx, Sy) * padValue;

% Place img1 in the center of imgTemp
Sox = round((Sx - Sx1) / 2) + 1;
Soy = round((Sy - Sy1) / 2) + 1;
imgTemp(Sox:Sox+Sx1-1, Soy:Soy+Sy1-1) = img1;

% Extract the region of target size (Sx2, Sy2) from the center
Sox = round((Sx - Sx2) / 2) + 1;
Soy = round((Sy - Sy2) / 2) + 1;
img2 = imgTemp(Sox:Sox+Sx2-1, Soy:Soy+Sy2-1);

end
