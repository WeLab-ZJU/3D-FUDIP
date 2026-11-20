function R = AddFrameletArray3D(A, B)
%ADDFRAMELETARRAY3D Add two multi-level 3D framelet coefficient trees.
%
%   R = AddFrameletArray3D(A, B)
%
%   A and B must have the same multi-level structure:
%   - A{ell} and B{ell} are 3D cell arrays of subbands.
%   - Every subband array has matching numeric size.

    narginchk(2,2);

    if ~iscell(A) || ~iscell(B) || numel(A) ~= numel(B)
        error('A and B must be cell arrays with the same number of levels.');
    end

    R = cell(size(A));

    for ell = 1:numel(A)
        if ~isequal(size(A{ell}), size(B{ell}))
            error('A{ell} and B{ell} must have the same subband layout at every level.');
        end

        R{ell} = cell(size(A{ell}));
        for idx = 1:numel(A{ell})
            R{ell}{idx} = A{ell}{idx} + B{ell}{idx};
        end
    end
end
