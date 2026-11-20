function psf = o2p(otf)
outSize = size(otf);
psf = ifftn(otf);
psf = circshift(psf,floor(outSize/2));
end