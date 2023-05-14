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

%% Mean_shift���࣬��ÿ������������Զ��������
% 
% k = 6; % 27���㣬������ ��template��Ϊ6
% Data_subpixel = [edges.x,edges.y]; % Data_subpixel��27�����������������
% 
% % ����
% % [centroid, point_classify_result] = Mean_Shift(Data_subpixel, 50); % centroid���������ģ� point_classify_result���������� 25������뾶��50��template
% [centroid, point_classify_result] = meanshiftcluster(Data_subpixel, 50);
% % ��Բ��ϼ���������� Points
% Points(k,2) = 0;
% for i = 1 : k
%     params = FitEllipse(Data_subpixel(point_classify_result==i,1), Data_subpixel(point_classify_result==i,2));
%     Points(i,1) = params(1);
%     Points(i,2) = params(2);
% end

%%

%% ��ԭͼ�ϻ��Ƴ����ؼ��������ؼ���Ե��
%Show image
figure(3);
% original
imshow(image,'InitialMagnification', 'fit'), hold on

seg = 0.6;
quiver(edges.x-seg/2*edges.ny, edges.y+seg/2*edges.nx, ...
    seg*edges.ny, -seg*edges.nx, 0, 'y.','linewidth',2);
% % ���ƾ�����
% PlotData(Data_subpixel, point_classify_result, Points);
% I1 = ShowEnlargedImageObject(I,[163,311],[197,348],3,2,2);
% imshow(I1);
toc