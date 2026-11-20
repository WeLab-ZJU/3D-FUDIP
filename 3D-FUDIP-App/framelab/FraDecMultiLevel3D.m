function Dec = FraDecMultiLevel3D(A, D, L)
%FRADECMULTILEVEL3D Multi-level tensor-product framelet decomposition in 3D.
%
%   Dec = FraDecMultiLevel3D(A, D, L)
%
%   A must be a 3D array. D is the 1D decomposition filter bank generated
%   by `GenerateFrameletFilter3D`. The output Dec is a level-wise cell array
%   whose entries are 3D subband cell arrays.

    Dec = cell(1, L);

    for k = 1:L
        Dec{k} = FraDec3D(A, D, k);
        A = Dec{k}{1, 1, 1};
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Dec = FraDec3D(A, D, L)
%FRADEC3D Single-level tensor-product framelet decomposition in 3D.

    nD = length(D);
    SorAS = D{nD};
    Dec = cell(nD - 1, nD - 1, nD - 1);

    for i = 1:nD-1
        M1 = D{i};
        temp1 = ipermute(ConvSymAsym3D(permute(A, [2 1 3]), M1, SorAS(i), L), [2 1 3]);

        for j = 1:nD-1
            M2 = D{j};
            temp2 = ConvSymAsym3D(temp1, M2, SorAS(j), L);

            for k = 1:nD-1
                M3 = D{k};
                Dec{i, j, k} = ipermute(ConvSymAsym3D(permute(temp2, [3 2 1]), M3, SorAS(k), L), [3 2 1]);
            end
        end
    end
end