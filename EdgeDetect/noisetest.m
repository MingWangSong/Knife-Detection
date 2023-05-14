clc
clear all
close all

%% load image

image = imread('./picture/4noise.tif');

BlurGaussian = 0.9;
G = fspecial('Gaussian',5,BlurGaussian);
img = imfilter(image,G,'replicate');

BlurQuantization = 1/sqrt(12);
BlurWidth = sqrt(BlurQuantization^2 + BlurGaussian^2);
[E] = edge(img,'canny',0.71);

%% Edgedetect with paper

threshold = 10;
iter = 3;
[edges1, RI] = subpixelEdges(img, threshold, 'SmoothingIter', iter); 

%% Edgedetect with Zernike 5x5
%使用高斯核模糊原始图像，模糊宽度（标准偏差）为0.75像素

edges2 = SubpixelEdge_5x5(img,E,BlurWidth);

%% Zernike 7x7

edges3 = ZernikeSubpixelEdgeDetect(im2double(img),E,BlurWidth);

%% cluster
k = 6; % 27个点，聚类数 ；template改为6
Data_subpixel1 = [edges1.x,edges1.y]; 
Data_subpixel2 = [edges2.u,edges2.v];
Data_subpixel3 = [edges3.u,edges3.v];

% paper聚类
[centroid1, point_classify_result1] = Mean_Shift(Data_subpixel1, 40); % centroid：聚类中心； point_classify_result：分类结果； 25：聚类半径；50：template
Points1(k,2) = 0;
for i = 1 : k
    params = FitEllipse(Data_subpixel1(point_classify_result1==i,1), Data_subpixel1(point_classify_result1==i,2));
    Points1(i,1) = params(1);
    Points1(i,2) = params(2);
end

% zernike 5x5聚类
[centroid2, point_classify_result2] = Mean_Shift(Data_subpixel2, 45); % centroid：聚类中心； point_classify_result：分类结果； 25：聚类半径；50：template
Points2(k,2) = 0;
for i = 1 : k
    params = FitEllipse(Data_subpixel2(point_classify_result2==i,1), Data_subpixel2(point_classify_result2==i,2));
    Points2(i,1) = params(1);
    Points2(i,2) = params(2);
end

% zernike 7x7聚类
[centroid3, point_classify_result3] = Mean_Shift(Data_subpixel3, 45); % centroid：聚类中心； point_classify_result：分类结果； 25：聚类半径；50：template
Points3(k,2) = 0;
for i = 1 : k
    params = FitEllipse(Data_subpixel3(point_classify_result3==i,1), Data_subpixel3(point_classify_result3==i,2));
    Points3(i,1) = params(1);
    Points3(i,2) = params(2);
end

%% Computer CenterBias
Originaldata = [
                40,45;
                50,151;
                130,49;
                148,140;
                222,150;
                250,55];
[dis1,mean1, var1] = CenterBias(Originaldata, Points1);
[dis2,mean2, var2] = CenterBias(Originaldata, Points2);
[dis3,mean3, var3] = CenterBias(Originaldata, Points3);
%% show image
figure(1);
imshow(image), hold on

seg = 0.6;
quiver(edges1.x-seg/2*edges1.ny, edges1.y+seg/2*edges1.nx, ...
    seg*edges1.ny, -seg*edges1.nx, 0, 'r.');
quiver(real(edges2.u)-seg/2*edges2.nv, real(edges2.v)+seg/2*edges2.nu, ...
    seg*edges2.nv, -seg*edges2.nu, 0, 'g.');
quiver(real(edges3.u)-seg/2*edges3.nv, real(edges3.v)+seg/2*edges3.nu, ...
    seg*edges3.nv, -seg*edges3.nu, 0, 'b.');