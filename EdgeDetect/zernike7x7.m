clc
clear all
close all

%% Step 1 Build a circle
%         
%---------------------------------------------------------------
tic
image = imread('./picture/noise.tif');
if numel(size(image)) == 3
    image = rgb2gray(image);
end
img = im2double(image);

%% Step 2 EdgeDetect

%ʹ�ø�˹��ģ��ԭʼͼ��ģ����ȣ���׼ƫ�Ϊ0.75����
BlurGaussian = 0.75;
G = fspecial('Gaussian',5,BlurGaussian);
img = imfilter(img,G,'replicate');
BlurQuantization = 1/sqrt(12);
BlurWidth = sqrt(BlurQuantization^2 + BlurGaussian^2);

[E] = edge(img,'canny',0.71);
% edges = SubpixelEdge_7x7(img,E,BlurWidth);
edges = ZernikeSubpixelEdgeDetect(img,E,BlurWidth);


k = 6; % 27���㣬������ ��template��Ϊ6
Data_subpixel = [edges.u,edges.v]; % Data_subpixel��27�����������������

% ����
[centroid, point_classify_result] = Mean_Shift(Data_subpixel, 45); % centroid���������ģ� point_classify_result���������� 25������뾶��50��template

% ��Բ��ϼ���������� Points
Points(k,2) = 0;
for i = 1 : k
    params = FitEllipse(Data_subpixel(point_classify_result==i,1), Data_subpixel(point_classify_result==i,2));
    Points(i,1) = params(1);
    Points(i,2) = params(2);
end

%Show image
figure(2);
falseimg = false(256,256);
imshow(falseimg), hold on
% imshow(img), hold on, axis on
% xlabel('v-direction'), ylabel('u-direction')
% plot(edges.u,edges.v,'w.')
scatter(edges.u,edges.v,50,'w.');
toc
%Display legend
% legend('True edge location','Pixel-level edge (Sobel)','Subpixel edge')

% imshow(img);hold on
% seg = 0.6;
% quiver(real(edges.u)-seg/2*edges.nv, real(edges.v)+seg/2*edges.nu, ...
%     seg*edges.nv, -seg*edges.nu, 0, 'r.');
% 
% 
% %% display normal vectors
% 
% quiver(edges.u, edges.v, edges.nu, edges.nv, 0, 'b');
% 
% 
% hold off