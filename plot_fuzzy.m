

close all; 
clear all; 
clc;



%% ��������
% ��(a,b,c)Ϊ���ģ�RΪ�뾶
a = 0; b = 0; c = 0;
R = 2;
[x, y, z] = sphere(50);
    % �����뾶
    x = R*x; 
    y = R*y;
    z = R*z;

    % ��������
    x = x+a;
    y = y+b;
    z = z+c;

figure(1)
plot3(x, y, z,'-')
hold on; 
grid on;

% 
% height = 80;
% timePoint = 500;
% % trailsNum = length(y_1)/timePoint;
% trailsNumlimit = 1:100 %500:700;
% 
% plot_p = one_trail_smooth(:, :);
% [x y]= meshgrid(1:timePoint, trailsNumlimit);
% z = sqrt();
% figure;
% surfcf(x, y, z);
% grid on;



