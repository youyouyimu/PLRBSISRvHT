clc;
clear;
close all;

scale = 2 ;
load(['parameters1\parameter_' num2str(scale)]);
load(['parameters1\dt_' num2str(scale)]);

% image=im2double(imread('Test\Set14\zebra.bmp'));
% image=im2double(imread('Test\Set14\ppt3.bmp'));
% image=im2double(imread('Test\Set14\pepper.bmp'));
% image=im2double(imread('Test\Set14\monarch.bmp'));
% image=im2double(imread('Test\Set14\man.bmp'));
% image=im2double(imread('Test\Set14\lenna.bmp'));
% image=im2double(imread('Test\Set14\foreman.bmp'));
% image=im2double(imread('Test\Set14\flowers.bmp'));
% image=im2double(imread('Test\Set14\face.bmp'));
% image=im2double(imread('Test\Set14\comic.bmp'));
% image=im2double(imread('Test\Set14\coastguard.bmp'));
% image=im2double(imread('Test\Set14\bridge.bmp'));
% image=im2double(imread('Test\Set14\barbara.bmp'));
image=im2double(imread('Test\Set14\baboon.bmp'));
 
% image=im2double(imread('Test\Set5\baby_GT.bmp'));
% image=im2double(imread('Test\Set5\bird_GT.bmp'));
% image=im2double(imread('Test\Set5\butterfly_GT.bmp'));
% image=im2double(imread('Test\Set5\head_GT.bmp'));
% image=im2double(imread('Test\Set5\woman_GT.bmp'));

image  = modcrop(image,scale);

h = fspecial('gaussian', 5, 1.6);  % the Gaussian filter
image_gauss = imfilter( image, h);

H_15 = [2 8 3 12 10 1 4 11 14 6 9 15 7 13 5];   % the sequence

sz1 = size(image);

if(size(sz1,2)==2)  
    imageL = imresize(image_gauss,1/scale,'bicubic');
    imageB = imresize(imageL,scale,'bicubic');
else
    image_ycbcr = rgb2ycbcr(image_gauss);
    
    image_y  = im2double(image_ycbcr(:,:,1));
    image_cb = im2double(image_ycbcr(:,:,2));
    image_cr = im2double(image_ycbcr(:,:,3));
    
    imageL    = imresize(image_y,1/scale,'bicubic');
    imageL_cb = imresize(image_cb,1/scale,'bicubic');
    imageL_cr = imresize(image_cr,1/scale,'bicubic');
    
    imageB = zeros(size(image_ycbcr));
    imageB(:,:,1) = imresize(imageL,scale,'bicubic');
    imageB(:,:,2) = imresize(imageL_cb,scale,'bicubic');
    imageB(:,:,3) = imresize(imageL_cr,scale,'bicubic');
    
    imageH_rec = zeros(size(image_ycbcr));
    imageH_rec(:,:,2) = imageB(:,:,2);
    imageH_rec(:,:,3) = imageB(:,:,3);
end

H_16=hadamard( 16 ); % the Hadamard matrix
H_16(:,1) =[]; 

sz = size(imageL);
imagepadding = zeros(sz(1)+2,sz(2)+2);  % image padding
imagepadding(2:end-1,2:end-1) = imageL;

offset = floor( scale / 2 );
    
startt = tic;   % the proposed SR method
[imageH]= SR_2_Hadamard( imagepadding, parameters, dt, H_16 );
toc(startt);

if(size(sz1,2)==2)
    imageH_rec = imageH;
else  
    imageH_rec(:,:,1) = imageH;
    imageB = ycbcr2rgb( imageB );
    imageH_rec = ycbcr2rgb( imageH_rec );
end

    
  if(mod(scale,2) == 0)
      if(size(sz1,2)==2)
          imageB = imageB( offset + scale + 2 + 3 : end - ( offset + scale + 1  + 3 ),...
              offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ));
          imageH_rec = imageH_rec( offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),...
              offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ));
          image  = image(offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),...
              offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ));
      else
          imageB = imageB( offset + scale + 2 + 3 : end - ( offset + scale + 1  + 3 ),...
              offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),:);
          imageH_rec = imageH_rec( offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),...
              offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),:);
          image  = image(offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),...
              offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),:);
      end
      
      [p1,s1] = compute_psnr_ssim(image,imageB); % Bicubic
      [p2,s2] = compute_psnr_ssim(image,imageH_rec); % Our

  else
      if(size(sz1,2)==2)
          imageB = imageB( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
              offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ) );
          imageH_rec = imageH_rec( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
              offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ) );
          image  = image( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
              offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ) );
      else
          imageB = imageB( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
              offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),:);
          imageH_rec = imageH_rec( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
              offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),:);
          image  = image( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
              offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),:);
      end
      
      [p1,s1] = compute_psnr_ssim(image,imageB); % Bicubic
      [p2,s2] = compute_psnr_ssim(image,imageH_rec); % Our

  end

figure('NumberTitle', 'off', 'Name', 'Bicubic');
imshow(imageB,'Border','tight');
figure('NumberTitle', 'off', 'Name', 'Our Propose Method');
imshow(imageH_rec,'Border','tight');

display(['Bicubic PSNR ' num2str(p1)]);
display(['Our     PSNR ' num2str(p2)]);
display(['Bicubic SSIM ' num2str(s1)]);
display(['Our     SSIM ' num2str(s2)]);

