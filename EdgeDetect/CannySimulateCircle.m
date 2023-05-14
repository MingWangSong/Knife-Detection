clc
clear all
close all

%% Canny
image = imread('./picture/BinaryCircle.tif');
Edge_Canny = edge(image, 'Canny');
imshow(Edge_Canny);
