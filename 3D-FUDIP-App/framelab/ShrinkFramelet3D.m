function R = ShrinkFramelet3D(A, tau)
%SHRINKFRAMELET3D Soft-threshold a multi-level 3D framelet coefficient tree.
%
%   R = ShrinkFramelet3D(A, tau)
%
%   The low-pass coefficient at each level, i.e. subband (1,1,1), is kept
%   unchanged, matching the behavior of the original 2D implementation.
%   All remaining subbands are soft-thresholded with threshold tau.
%
%   The shrinkage rule is magnitude-based and therefore works for both real
%   and complex coefficients.

    narginchk(2,2);
    validateattributes(tau, {'numeric'}, {'real','scalar','nonnegative'}, mfilename, 'tau', 2);

    if ~iscell(A)
        error('A must be a multi-level framelet coefficient cell array.');
    end

    R = cell(size(A));

    for ell = 1:numel(A)
        subSize = size(A{ell});
        if numel(subSize) ~= 3
            error('A{ell} must be a 3D cell array of subbands.');
        end

        R{ell} = cell(subSize);

        for idx = 1:numel(A{ell})
            [i,j,k] = ind2sub(subSize, idx);
            coeff   = A{ell}{idx};

            if i == 1 && j == 1 && k == 1
                R{ell}{idx} = coeff;
            else
                tauLike = cast(tau, 'like', coeff);
                mag   = abs(coeff);
                scale = max(mag - tauLike, 0);
                denom = mag;
                denom(denom == 0) = 1;
                R{ell}{idx} = scale .* (coeff ./ denom);
            end
        end
    end
end
