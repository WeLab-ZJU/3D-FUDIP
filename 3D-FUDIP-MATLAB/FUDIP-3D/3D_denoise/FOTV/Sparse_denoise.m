function SHVideo = Sparse_denoise(inputImg)

warning('off');
addpath('./Sparse_denoise/SHOperation');
addpath('./Sparse_denoise/SHIter');


gpu = 1;
SHIter = 1;
fidelity= 100 ;
sparsity= 8;
% lambda=settingRECON.lambda;
% lambda=lambda*10^-9;
% Pixel=settingRECON.Pixel ;
% Pixel=Pixel*10^-9;
zconti= 0.1;

% Background = (settingRECON.Background);
Oversample = 3;

%%%%%%%%%%%%%%%%%%% Load Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% do not need read tiff file 
SIMmovie=inputImg;
% SIMmovie=single(SIMmovie);
% constant=max(SIMmovie(:));
% SIMmovie=SIMmovie./constant;

if size(SIMmovie,3)<3
    flage3=size(SIMmovie,3);
    SIMmovie=padarray(SIMmovie,[0,0,3-size(SIMmovie,3)],'replicate' );
else
    flage3=size(SIMmovie,3);
end

%%%%%%%%%%%% Check data size  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VIDEOResolution = [size(SIMmovie,1)  size(SIMmovie,2)  size(SIMmovie,3)];
% disp(['Data size is ' num2str(VIDEOResolution(1)) ' X ' num2str(VIDEOResolution(2)) ' X ' num2str(VIDEOResolution(3))]);
%%%%%%%%%%%%% RUN Reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% SIMmovie=SIMmovie./max(SIMmovie(:));

% 
% switch Background
%     case 1
%         backgrounds = background_estimation(SIMmovie./2);
%         SIMmovie = SIMmovie-backgrounds;
%     case 2
%         backgrounds= background_estimation(SIMmovie./2.5);
%         SIMmovie = SIMmovie-backgrounds;
%     case 3
%         medVal = mean(SIMmovie(:));
%         sub_temp = SIMmovie;
%         sub_temp(sub_temp > medVal)=medVal;
%         backgrounds= background_estimation(sub_temp);
%         SIMmovie = SIMmovie-backgrounds;
%     case 4
%         medVal = mean(SIMmovie(:))./2;
%         sub_temp = SIMmovie;
%         sub_temp(sub_temp > medVal)=medVal;
%         backgrounds = background_estimation(sub_temp);
%         SIMmovie = SIMmovie-backgrounds;
%     case 5
%         medVal = mean(SIMmovie(:))./2.5;
%         sub_temp = SIMmovie;
%         sub_temp(sub_temp > medVal)=medVal;
%         backgrounds = background_estimation(sub_temp);
%         SIMmovie = SIMmovie - backgrounds;
%     case 6
%         ;
% end

SIMmovie(SIMmovie < 0)=0;
% SIMmovie = SIMmovie./max(SIMmovie(:));

switch Oversample
    case 1
        y=Spatial_Oversample(SIMmovie);
        f=single(y);
    case 2
        y=Fourier_Oversample(SIMmovie);
        f=single(y);
    case 3
%         f=single(SIMmovie);
        f = SIMmovie;
end

disp(['Sparse reconstruction is ongoing...']);
SHVideo=SparseHessian_core(f,fidelity,zconti,sparsity,SHIter,gpu);
SHVideo=SHVideo(:,:,1:flage3);
end