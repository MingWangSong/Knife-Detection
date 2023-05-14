clc
clear all
close all

%% Step 1 load image
image = imread('./picture/point.tif');
image1 = imread('./picture/point-preprocess.tif');
if numel(size(image)) == 3
    image = rgb2gray(image);
end

img = im2double(image);

%% Step 2 EdgeDetect

%使用高斯核模糊原始图像，模糊宽度（标准偏差）为0.75像素
BlurGaussian = 0.9;
% G = fspecial('Gaussian',5,BlurGaussian);
% img = imfilter(img,G,'replicate');
BlurQuantization = 1/sqrt(12);
BlurWidth = sqrt(BlurQuantization^2 + BlurGaussian^2);

[E] = edge(img,'canny',0.71);
% [E] = edge(img,'canny');
threshold = 5;
iter = 3;


edges3 = SubpixelEdge_5x5(img,E,BlurWidth);

% edges2 = SubpixelEdge_7x7(img,E,BlurWidth);

edges2 = ZernikeSubpixelEdgeDetect(img,E,BlurWidth);

[edges1, RI] = subpixelEdges(image1, threshold, 'SmoothingIter', iter); 

%%
%% cluster
k = 27; % 27个点，聚类数 ；template改为6
Data_subpixel1 = [edges1.x,edges1.y]; 
Data_subpixel2 = [edges2.u,edges2.v];
Data_subpixel3 = [edges3.u,edges3.v];

% paper聚类
[centroid1, point_classify_result1] = Mean_Shift(Data_subpixel1, 25); % centroid：聚类中心； point_classify_result：分类结果； 25：聚类半径；50：template
Points1(k,2) = 0;
for i = 1 : k
    params = FitEllipse(Data_subpixel1(point_classify_result1==i,1), Data_subpixel1(point_classify_result1==i,2));
    Points1(i,1) = params(1);
    Points1(i,2) = params(2);
end

% zernike 5x5聚类
[centroid2, point_classify_result2] = Mean_Shift(Data_subpixel2, 25); % centroid：聚类中心； point_classify_result：分类结果； 25：聚类半径；50：template
Points2(k,2) = 0;
for i = 1 : k
    params = FitEllipse(Data_subpixel2(point_classify_result2==i,1), Data_subpixel2(point_classify_result2==i,2));
    Points2(i,1) = params(1);
    Points2(i,2) = params(2);
end

% zernike 7x7聚类
[centroid3, point_classify_result3] = Mean_Shift(Data_subpixel3, 25); % centroid：聚类中心； point_classify_result：分类结果； 25：聚类半径；50：template
Points3(k,2) = 0;
for i = 1 : k
    params = FitEllipse(Data_subpixel3(point_classify_result3==i,1), Data_subpixel3(point_classify_result3==i,2));
    Points3(i,1) = params(1);
    Points3(i,2) = params(2);
end

%% Show Result

figure(1);

imshow(image,'InitialMagnification', 'fit'), hold on

% scatter(edges1.x,edges1.y,50,'r.');
% scatter(edges2.u,edges2.v,50,'g.');
% scatter(edges3.u,edges3.v,50,'b.');
seg = 0.6;
quiver(edges1.x-seg/2*edges1.ny, edges1.y+seg/2*edges1.nx, ...
    seg*edges1.ny, -seg*edges1.nx, 0, 'r.','linewidth',2);
% quiver(real(edges2.u)-seg/2*edges2.nv, real(edges2.v)+seg/2*edges2.nu, ...
%     seg*edges2.nv, -seg*edges2.nu, 0, 'g.');
quiver(real(edges3.u)-seg/2*edges3.nv, real(edges3.v)+seg/2*edges3.nu, ...
    seg*edges3.nv, -seg*edges3.nu, 0, 'b.','linewidth',2);