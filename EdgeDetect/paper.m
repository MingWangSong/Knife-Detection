clc
clear all
close all

%% Step 1 Build a circle
tic
image = imread('./picture/point.tif');
% image1 = imread('./picture/point-preprocess.tif');
% original
% image = imread('./picture/1-3.tif');
if numel(size(image)) == 3
    image = rgb2gray(image);
end
         
%% subpixel detection
threshold = 5;
iter = 3;
[edges, RI] = subpixelEdges(image, threshold, 'SmoothingIter', iter); 

%% show image
% showRestoredImage = true;
% if showRestoredImage
%     imshow(RI/255,'InitialMagnification', 'fit');
% else
%     imshow(image,'InitialMagnification', 'fit');
% end
% 
% visEdges(edges);
% figure(3);
% falseimg = false(256,256);
% imshow(falseimg), hold on
% % imshow(img), hold on, axis on
% % xlabel('v-direction'), ylabel('u-direction')
% % plot(edges.u,edges.v,'w.')
% scatter(edges.x,edges.y,50,'w.');

%% Mean_shift聚类，将每个插针的坐标自动分离出来
% 
% k = 6; % 27个点，聚类数 ；template改为6
% Data_subpixel = [edges.x,edges.y]; % Data_subpixel：27个插针的亚像素坐标
% 
% % 聚类
% % [centroid, point_classify_result] = Mean_Shift(Data_subpixel, 50); % centroid：聚类中心； point_classify_result：分类结果； 25：聚类半径；50：template
% [centroid, point_classify_result] = meanshiftcluster(Data_subpixel, 50);
% % 椭圆拟合计算插针中心 Points
% Points(k,2) = 0;
% for i = 1 : k
%     params = FitEllipse(Data_subpixel(point_classify_result==i,1), Data_subpixel(point_classify_result==i,2));
%     Points(i,1) = params(1);
%     Points(i,2) = params(2);
% end

%%

%% 在原图上绘制出像素级和亚像素级边缘点
%Show image
figure(3);
% original
imshow(image,'InitialMagnification', 'fit'), hold on

seg = 0.6;
quiver(edges.x-seg/2*edges.ny, edges.y+seg/2*edges.nx, ...
    seg*edges.ny, -seg*edges.nx, 0, 'y.','linewidth',2);
% % 绘制聚类结果
% PlotData(Data_subpixel, point_classify_result, Points);
% I1 = ShowEnlargedImageObject(I,[163,311],[197,348],3,2,2);
% imshow(I1);
toc