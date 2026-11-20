function [w1,b1] = update_shearlet(X,w1,b1,lamda,alpha,scales)

useGPU = isa(X,'gpuArray');
[Xfreq, ~, preparedFilters, dualFrameWeightsCurr, shearletIdxs] = SLprepareSerial3D(useGPU,X,scales);

if useGPU == 0
    
    for j = 1:size(shearletIdxs,1)
        shearletIdx = shearletIdxs(j,:);
        %%shearlet decomposition
        [coeffs,~, dualFrameWeightsCurr,~] = SLsheardecSerial3D(Xfreq,shearletIdx,preparedFilters,dualFrameWeightsCurr);
        coeffs = gather(coeffs);     
        w1(:,:,:,j) = shrink_operator(coeffs,b1(:,:,:,j),lamda./alpha);
        b1(:,:,:,j) = b1(:,:,:,j) + coeffs - w1(:,:,:,j);    
    end
    
elseif useGPU ==1
    
     for j = 1:size(shearletIdxs,1)
        b = gpuArray(b1(:,:,:,j)); 
        shearletIdx = shearletIdxs(j,:);
        %%shearlet decomposition
        [coeffs,~, dualFrameWeightsCurr,~] = SLsheardecSerial3D(Xfreq,shearletIdx,preparedFilters,dualFrameWeightsCurr);    
        w = shrink_operator(coeffs,b,lamda./alpha);
        b = b + coeffs - w;
        clear coeffs 
        w1(:,:,:,j) = gather(w);
        b1(:,:,:,j) = gather(b);    
    end
   
end
 
end 