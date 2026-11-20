clear 
i = loadtiff("D:\code\Julia\ThreeDeconv.jl\example\datasetwithraw\raw\cell\raw\cell_fitc.tif");
% f = loadtiff('E:\data\hlm\denonv_test2\denonv_test2_crop_WBdenonv.tif');
f = i(500:1500,500:1500,1:30);
f = single(f);
f = f./max(f(:));
raw = f;

f = gpuArray(f);
useGPU = 1;
fidelity = 100;
mu = 0.1;
paral1 = 0;
iter_num = 100;
contiz = 0.4;


g = FOTV_Hessian_denoise(f,fidelity,mu,paral1,iter_num,contiz,useGPU);

g = gather(g);
subplot(121),imshow(raw(:,:,13),[])
subplot(122),imshow(g(:,:,13),[])