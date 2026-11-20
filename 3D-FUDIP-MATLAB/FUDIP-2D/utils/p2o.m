function otf = p2o(psf)
%   Convert PSF to OTF.
%   OTF = p2o(PSF) converts a point-spread function (PSF) to its
%   optical transfer function (OTF) using FFT.
%
%   Assumption:
%   - size(psf) is already equal to the target blurred image size.
%   - psf center is defined consistently with MATLAB psf2otf, i.e.
%     shifted by -floor(size(psf)/2).
%
%   This is a simplified version of `psf2otf`.

% basic validation
    if ~isnumeric(psf) || issparse(psf)
        error('p2o:InvalidInput', 'PSF must be numeric and non-sparse.');
    end
    if ~all(isfinite(psf(:)))
        error('p2o:InvalidInput', 'PSF must contain only finite values.');
    end

    psfSize = size(psf); 

    % move PSF center to (1,1) before FFT
    psf = circshift(psf,-floor(psfSize/2));

    % FFT to obtain OTF
    otf = fftn(psf);
end
