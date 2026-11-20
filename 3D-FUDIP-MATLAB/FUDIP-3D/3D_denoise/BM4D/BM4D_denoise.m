function img = BM4D_denoise(img,sigma)
% parallel denoising

% parfor frame = 1:size(img,3)    
%     [res(:,:,frame), sigma(frame)] = modified_ACSN_core(img(:,:,frame),para); 
% end

% figure(),
% subplot(121),imshow(img(:,:,10),[])
% subplot(122),imshow(res(:,:,10),[])

tic
psd = mean(sigma).*ones(8);   
sType = 2;

if sum(size(img)>[256 256 20])  % [256 256 20]
    size_y = min(200,size(img,1));
    size_x = min(200,size(img,2));
    size_z = min(100,size(img,3));
    overlap = 5;
            
    Tiles = im2tiles(img,overlap,size_x,size_y,size_z);
    parfor idx = 1:numel(Tiles)
         Tiles{idx} = Video_filtering_t(Tiles{idx},psd,psd,'dct',0,1,sType,'np',0);
    end
    img = tiles2im(Tiles,overlap);
    clear Tiles;        
else
            
    img = Video_filtering_t(img,psd,psd,'dct',0,1,sType,'np',0);
            
end
        
parfor i = 1:size(img,3)
    M1 = max(max(img(:,:,i)));
    M2 = min(min(img(:,:,i)));
    img(:,:,i) = (img(:,:,i) - M2)./(M1 -M2);
    img(:,:,i) = Video_filtering_xy(img(:,:,i),sigma);
    img(:,:,i) = (img(:,:,i)).*(M1-M2)+ M2;
            
end
toc
end