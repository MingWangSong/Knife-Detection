%%
clear;
clc;

image = imread('./picture/point.tif');
% thresh = graythresh(image);

ima = imbinarize(image,0.5);
% imb = imbinarize(image,0.5);
% imc = imbinarize(image,0.6);
figure(1);
imshow(ima);
% figure(2);
% imshow(imb);
% figure(3);
% imshow(imc);
imb = double(ima);
for i = 1:360
    if i<31 || i>45&&i<170 || i>186&&i<310 || i>325
        imb(i,:) = 0;
    end
end
for j = 1:660
    if j>40&&j<100 || j>192&&j<245 || j>415&&j<465 || j>557&&j<613 || j>630
        imb(:,j) = 0;
    end
end
figure(2);
imshow(imb);

%% 
% clear all;
% f=imread('./picture/point.tif');
% % f=rgb2gray(f);%转换为灰度图像
% f=im2double(f);%数据类型转换
% %全剧阈值
% T=0.5*(min(f(:))+max(f(:)));
% done=false;
% while ~done
% 	g=f>=T;
% 	Tn=0.5*(mean(f(g))+mean(f(~g)));
% 	done = abs(T-Tn)<0.1;
% 	T=Tn;
% end
% display('Threshold(T)-Iterative');%显示文字
% T
% r=im2bw(f,T);
% subplot(221);imshow(f);
% xlabel('(a)原始图像');
% subplot(222);imshow(r);
% xlabel('(b)迭代法全局阈值分割');
% Th=graythresh(f);%阈值
% display('Global Thresholding- Otsu''s Method');
% Th
% s=im2bw(f,Th);
% subplot(223);imshow(s);
% xlabel('(c)全局阈值Otsu法阈值分割');
% se=strel('disk',10);
% ft=imtophat(f,se);
% Thr=graythresh(ft);
% display('Threshold(T) -Local Thresholding');
% Thr
% lt = im2bw(ft,Thr);
% subplot(224);imshow(lt);
% xlabel('(d)局部阈值分割');