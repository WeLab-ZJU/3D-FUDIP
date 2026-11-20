function [sum_c,sum_s] = get_shearlet(w1,b1,scales)

X = gpuArray(w1(:,:,:,1)-b1(:,:,:,1));
useGPU = isa(X,'gpuArray');
[Xfreq, ~, preparedFilters, dualFrameWeightsCurr, shearletIdxs] = SLprepareSerial3D(useGPU,X,scales);

clear X

sum_c = gpuArray(zeros(size(Xfreq)));
sum_s = gpuArray(zeros(size(Xfreq)));

 for j = 1:size(shearletIdxs,1)
    shearletIdx = shearletIdxs(j,:);
    Q = gpuArray(w1(:,:,:,j)-b1(:,:,:,j));
    
    Qfreq = fftshift(fftn(ifftshift(Q)));
    clear Q
    %%shearlet decomposition
    [coeffs,shearlet, dualFrameWeightsCurr,~] = SLsheardecSerial3D(Qfreq,shearletIdx,preparedFilters,dualFrameWeightsCurr);
    sum_c = sum_c + coeffs;
    sum_s = sum_s + shearlet.*conj(shearlet);
    
    clear Qfreq
 end

end