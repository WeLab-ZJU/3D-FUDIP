function Rec = FraRecMultiLevel3D(C, R, L)
%FRARECMULTILEVEL3D Multi-level tensor-product framelet reconstruction in 3D.
%
%   Rec = FraRecMultiLevel3D(C, R, L)

    for k = L:-1:2
        C{k - 1}{1, 1, 1} = FraRec3D(C{k}, R, k);
    end

    Rec = FraRec3D(C{1}, R, 1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Rec = FraRec3D(C, R, L)
%FRAREC3D Single-level tensor-product framelet reconstruction in 3D.

    nR = length(R);
    SorAS = R{nR};
    ImSize = size(C{1, 1, 1});
    template = C{1, 1, 1};
    Rec = zeros(ImSize, 'like', template);

    for i = 1:nR-1
        M1 = R{i};
        temp1 = zeros(ImSize, 'like', template);

        for j = 1:nR-1
            M2 = R{j};
            temp2 = zeros(ImSize, 'like', template);

            for k = 1:nR-1
                M3 = R{k};
                temp2 = temp2 + ipermute(ConvSymAsym3D(permute(C{i, j, k}, [3 2 1]), M3, SorAS(k), L), [3 2 1]);
            end

            temp1 = temp1 + ConvSymAsym3D(temp2, M2, SorAS(j), L);
        end

        Rec = Rec + ipermute(ConvSymAsym3D(permute(temp1, [2 1 3]), M1, SorAS(i), L), [2 1 3]);
    end
end