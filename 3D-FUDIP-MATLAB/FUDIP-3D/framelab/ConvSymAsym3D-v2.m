function C = ConvSymAsym3D(A, M, b, L)
%CONVSYMASYM3D Dilated 1D correlation along the first dimension of a 3D array.
%
%   C = ConvSymAsym3D(A, M, b, L)
%
%   Inputs
%   ------
%   A : input 3D array.
%   M : 1D analysis/reconstruction filter.
%   b : boundary-condition flag
%         'c' = circular
%         's' = symmetric
%         'a' = antisymmetric
%   L : framelet level (controls the dilation step 2^(L-1)).
%
%   Output
%   ------
%   C : filtered array with the same size as A.
%
%   Notes
%   -----
%   1) The operation implemented here is correlation, not convolution.
%      This matches the behavior expected by the framelet code.
%   2) The filter is always applied along dimension 1. Use
%      ApplyFilterAlongDim3D to filter along dimension 2 or 3.
%   3) The circular branch is implemented without toolbox functions.

    narginchk(4,4);

    if ~isnumeric(A)
        error('Input A must be numeric.');
    end
    if ~isvector(M) || ~isnumeric(M)
        error('Filter M must be a numeric vector.');
    end
    validateattributes(L, {'numeric'}, {'real','scalar','integer','>=',1}, mfilename, 'L', 4);

    step = 2^(L - 1);
    ker  = zeros(step * (numel(M) - 1) + 1, 1, 'like', A);
    ker(1:step:end) = cast(M(:), 'like', A);
    padLen = floor(numel(ker) / 2);

    switch lower(b)
        case 'c'
            C = LocalCircularCorrelationDim1(A, ker);

        case {'s', 'a'}
            Ae = LocalPadSymAsymDim1(A, padLen, lower(b));
            C  = convn(Ae, reshape(flip(ker(:)), [], 1, 1), 'valid');

        otherwise
            error('Unknown boundary flag ''%s''. Use ''c'', ''s'', or ''a''.', b);
    end
end

%==========================================================================
function C = LocalCircularCorrelationDim1(A, ker)
%LOCALCIRCULARCORRELATIONDIM1 Circular correlation along dimension 1.

    center = floor(numel(ker) / 2) + 1;
    C      = zeros(size(A), 'like', A);

    for t = 1:numel(ker)
        wt = ker(t);
        if wt ~= 0
            shift = center - t;
            C = C + wt * circshift(A, [shift 0 0]);
        end
    end
end

%==========================================================================
function Ae = LocalPadSymAsymDim1(A, p, bc)
%LOCALPADSYMASYMDIM1 Symmetric/antisymmetric extension along dimension 1.
%
%   The extension matches the spirit of the original 2D implementation:
%   symmetric padding is created first; for the antisymmetric case, the
%   padded samples on both sides are multiplied by -1.

    if p == 0
        Ae = A;
        return;
    end

    m = size(A,1);
    idx = LocalSymmetricPadIndices(m, p);
    Ae  = A(idx,:,:);

    if bc == 'a'
        sgn = ones(numel(idx), 1, 'like', A);
        sgn(1:p) = -1;
        sgn(end-p+1:end) = -1;
        Ae = bsxfun(@times, Ae, reshape(sgn, [], 1, 1));
    end
end

%==========================================================================
function idx = LocalSymmetricPadIndices(m, p)
%LOCALSYMMETRICPADINDICES Return dimension-1 indices for symmetric padding.
%
%   For example, with m = 4 and p = 2, the index vector is
%   [2 1 1 2 3 4 4 3].

    base   = [1:m, m:-1:1];
    period = 2 * m;
    pos    = mod((-p:(m+p-1)), period) + 1;
    idx    = base(pos);
end
