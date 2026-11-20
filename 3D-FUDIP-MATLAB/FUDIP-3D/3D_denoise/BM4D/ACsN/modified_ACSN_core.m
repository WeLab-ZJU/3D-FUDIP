function [out, sigma] = modified_ACSN_core(I,sigma)
% OTF radius
NA = 0.2;
Lambda = .520;
PxSize = .065;
R = 2*NA/Lambda*PxSize*size(I,1);
adj = 1.1;
R2 = (.5*size(I,1)*adj);
% multiplicative factor to adjust the sigma of the noise
ratio = sqrt(R2/abs(R-R2));
% rescaling
I(I<=0) = 1e-6;
% sigma = 0.6;
%% Evaluation of sigma
%{
[Values, BinCenters] = hist(I(:));
bins = BinCenters;

[~, first_min] = min(Values);
a1_est = bins(round(first_min/2));
a0_est = max(Values);

fo = fitoptions('Method','NonlinearLeastSquares',...
    'StartPoint',[a0_est a1_est]);
ft = fittype('a0*exp(-(1/2)*((x)/a1)^2)','options',fo);
[curve] = fit(bins',Values',ft);

a = curve.a1;
w = 1; 
sigma = w*ratio*a;
%}
% sigma = NLEstimate(img(:,:,10)); 
%% normalization
M1 = max(max(I));
M2 = min(min(I));
I2 = (I - M2)./(M1 -M2);
%% Denoising
% scaling sigma for non-8 bit images
if (M1-M2)>255
    sigma = sigma/(M1-M2)*255;
end

[~, img0] = Sparse_filtering([],I2,sigma./3,'np');

out = (img0).*(M1-M2)+ M2;

end