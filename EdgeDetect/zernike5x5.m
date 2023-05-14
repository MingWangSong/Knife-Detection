clc
clear all
close all

%% Step 1 Build a circle
tic
% image = imread('./picture/33BinaryCircle.tif');
% if numel(size(image)) == 3
%     image = rgb2gray(image);
% end
image = imread('./picture/point.tif');
img = im2double(image);

%% Step 2 EdgeDetect

%使用高斯核模糊原始图像，模糊宽度（标准偏差）为0.75像素
BlurGaussian = 0.9;
% G = fspecial('Gaussian',5,BlurGaussian);
% img = imfilter(img,G,'replicate');
BlurQuantization = 1/sqrt(12);
BlurWidth = sqrt(BlurQuantization^2 + BlurGaussian^2);

[E] = edge(img,'canny',0.75);
% [E] = edge(img,'canny');
edges = SubpixelEdge_5x5(img,E,BlurWidth);

%% Show image
% % figure(2);
% % imshow(img), hold on, axis on
% % xlabel('v-direction'), ylabel('u-direction')
% % plot(edges.nu,edges.nv,'w.')
% % 
% % %Display legend
% % legend('True edge location','Pixel-level edge (Sobel)','Subpixel edge')
% 
% imshow(image,'InitialMagnification', 'fit'), hold on
% seg = 0.6;
% quiver(edges.u-seg/2*edges.nv, edges.v+seg/2*edges.nu, ...
%     seg*edges.nv, -seg*edges.nu, 0, 'r.');
% 
% % display normal vectors
% quiver(edges.u, edges.v, edges.nu, edges.nv, 0, 'b');
% hold off

%% Mean_shift聚类，将每个插针的坐标自动分离出来

k = 27; % 27个点，聚类数 ；template改为6
Data_subpixel = [edges.u,edges.v]; % Data_subpixel：27个插针的亚像素坐标

% 聚类
[centroid, point_classify_result] = Mean_Shift(Data_subpixel, 25); % centroid：聚类中心； point_classify_result：分类结果； 25：聚类半径；50：template

% 椭圆拟合计算插针中心 Points
Points(k,2) = 0;
for i = 1 : k
    params = FitEllipse(Data_subpixel(point_classify_result==i,1), Data_subpixel(point_classify_result==i,2));
    Points(i,1) = params(1);
    Points(i,2) = params(2);
end

%%
% figure(1);
% falseimg = false(256,256);
% imshow(falseimg), hold on
% imshow(img), hold on, axis on
% xlabel('v-direction'), ylabel('u-direction')
% plot(edges.u,edges.v,'w.')
% scatter(edges.u,edges.v,50,'w.');

%% 在原图上绘制出像素级和亚像素级边缘点
%Show image
% figure(1);
% % falseimg = false(256,256);
imshow(image), hold on
% % , axis on
% % xlabel('v-direction'), ylabel('u-direction')
% 
% %Overlay true edge location
% % plot(u_edge,v_edge,'r')
% 
% %Get pixel-level edges for plotting
% % [E_v,E_u]  = ind2sub( size(OriginalImg), E_idx ); 
% % plot(E_u,E_v,'gx')
% 
% %Plot subpxiel edges
% % plot(edges.u,edges.v,'w.')
% % scatter(edges.u,edges.v,50,'w.');
seg = 0.6;
quiver(edges.u-seg/2*edges.nv, edges.v+seg/2*edges.nu, ...
    seg*edges.nv, -seg*edges.nu, 0, 'r.');
% % 绘制聚类结果
% PlotData(Data_subpixel, point_classify_result, Points);
% 
% %Display legend
% % legend('True edge location','Pixel-level edge (Sobel)','Subpixel edge')

toc