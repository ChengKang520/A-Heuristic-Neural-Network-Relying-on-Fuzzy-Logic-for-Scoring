
clear all;
close all;
clc;

%% load data

load('Small.mat');

% using diferent operator to process the raw image, finally, 
%   the canny operator performed best
Sample_input1 = Small.Mass.Train.Input(:,:,1,21);

%% Different Operators for Edge Detection
figure;
subplot(1,2,1);
imshow(Sample_input1);
ED1 = edge(Sample_input1,'sobel');
subplot(1,2,2);
imshow(ED1);

ED2 = edge(Sample_input1,'roberts');
figure;
subplot(1,2,1);
imshow(ED2);
ED3 = edge(Sample_input1,'prewitt');
subplot(1,2,2);
imshow(ED3);

ED4 = edge(Sample_input1,'log');
figure;
subplot(1,2,1);
imshow(ED4);
ED5 = edge(Sample_input1,'canny');
subplot(1,2,2);
imshow(ED5);

%% Define Type Reduction Method --> Type 2 Fuzzy Logic
%  TRMethod='KM' -> 1, 'EKM'-> 2, 'IASC' -> 3, 'EIASC' -> 4, 'EODS' -> 5, 'WM' -> 6, 'NT' -> 7, 'BMM' -> 8 
TRMethod = 1;

I = double(Sample_input1);
% 
classType = class(Sample_input1);
scalingFactor = double(intmax(classType));
I = I/scalingFactor;
% 
Gx = [-1 1];
Gy = Gx';
Ix = conv2(I,Gx,'same');
Iy = conv2(I,Gy,'same');
% 
figure
image(Ix,'CDataMapping','scaled')
colormap('gray')
title('Ix')

figure
image(Iy,'CDataMapping','scaled')
colormap('gray')
title('Iy')

t2fis=readt2fis('imageProcessing.t2fis');

Ieval = zeros(size(I));
for ii = 1:size(I,1)
    disp(ii);
    for jj = 1:length(I)
        Ieval(ii,jj) = evalt2([(Ix(ii,jj));(Iy(ii,jj));]',t2fis,TRMethod); 
    end
end

figure
image(I,'CDataMapping','scaled')
colormap('gray')
title('Original Grayscale Image')

Ieval_fig = zeros(size(Ieval));
% Ieval_fig(find(Ieval >= mean(mean(Ieval)))) = 1;
% Ieval_fig(find(Ieval < mean(mean(Ieval)))) = 0;
Ieval_fig = uint8(Ieval);

figure
image(Ieval_fig,'CDataMapping','scaled')
colormap('gray')
title('Edge Detection Using Fuzzy Logic')







