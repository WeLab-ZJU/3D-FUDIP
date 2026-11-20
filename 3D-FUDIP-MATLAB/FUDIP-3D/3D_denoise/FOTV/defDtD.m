function out = defDtD(kernel,ksize)
out = fftn(kernel,ksize).*conj(fftn(kernel,ksize));
out = single(out);
% out = abs(psf2otf(kernel,ksize));
end