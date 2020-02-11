function surfcf(varargin)
% 绘制带填充等值线图的曲面图，相当于surf+contourf
 
    hold on
    % plot the surface
    surf(varargin{:});
    shading interp; %插值阴影模式
 
    % plot filled contour and get handle to hggroup object
    [C,h] = contourf(varargin{:});
    c = get(h, 'Children');
 
    % set the Z-data for each patch object to lower limit of Z-axis
    zmin = min(zlim);
    for i = 1:length(c)
        set(c(i), 'zdata', zmin*ones(size(get(c(i), 'xdata'))));
    end
    % 去掉等值线图的线条，只以填充颜色显示
    for ii = 1:length(h)
        set(h(ii), 'LineStyle', 'none');
    end
 
    view(-10, 30); %设置3d视角
 
end
 