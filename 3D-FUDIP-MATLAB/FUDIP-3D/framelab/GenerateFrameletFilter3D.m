function [D, R] = GenerateFrameletFilter3D(frame)
%GENERATEFRAMELETFILTER3D Generate 1D framelet filters for 3D tensor products.
%
%   [D, R] = GenerateFrameletFilter3D(frame)
%
%   This helper keeps the frame numbering consistent with the original
%   2D FUDIP implementation:
%       frame = 1 : Haar wavelet
%       frame = 2 : Piecewise linear framelet
%       frame = 3 : Piecewise cubic framelet
%
%   The returned 1D filters are used by the 3D tensor-product framelet
%   decomposition and reconstruction operators.

    narginchk(1, 1);

    switch frame
        case 1
            % Haar wavelet
            D{1} = [0 1 1] / 2;
            D{2} = [0 1 -1] / 2;
            D{3} = 'cc';

            R{1} = [1 1 0] / 2;
            R{2} = [-1 1 0] / 2;
            R{3} = 'cc';

        case 2
            % Piecewise linear framelet
            D{1} = [1 2 1] / 4;
            D{2} = [1 0 -1] / 4 * sqrt(2);
            D{3} = [-1 2 -1] / 4;
            D{4} = 'ccc';

            R{1} = [1 2 1] / 4;
            R{2} = [-1 0 1] / 4 * sqrt(2);
            R{3} = [-1 2 -1] / 4;
            R{4} = 'ccc';

        case 3
            % Piecewise cubic framelet
            D{1} = [1 4 6 4 1] / 16;
            D{2} = [1 2 0 -2 -1] / 8;
            D{3} = [-1 0 2 0 -1] / 16 * sqrt(6);
            D{4} = [-1 2 0 -2 1] / 8;
            D{5} = [1 -4 6 -4 1] / 16;
            D{6} = 'ccccc';

            R{1} = [1 4 6 4 1] / 16;
            R{2} = [-1 -2 0 2 1] / 8;
            R{3} = [-1 0 2 0 -1] / 16 * sqrt(6);
            R{4} = [1 -2 0 2 -1] / 8;
            R{5} = [1 -4 6 -4 1] / 16;
            R{6} = 'ccccc';

        otherwise
            error('Unsupported frame type %g. Use 1 (Haar), 2 (linear), or 3 (cubic).', frame);
    end
end
