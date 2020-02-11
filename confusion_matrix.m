 function confusion_matrix(actual,detected)
 [mat,order] = confusionmat(actual,detected);
 
 %mat = rand(10);           %# A 5-by-5 matrix of random values from 0 to 1
% mat(3,3) = 0;            %# To illustrate
% mat(5,2) = 0;            %# To illustrate
imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)
 
textStrings = num2str(mat(:),'%0.01f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
 
length_category = size(mat,1);
%% ## New code: ###
%idx = find(strcmp(textStrings(:), '0.00'));
%textStrings(idx) = {'   '};
%% ################
 
[x,y] = meshgrid(1:length_category);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center','color',[1 0 0]);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
 
set(gca,'XTick',1:length_category,...                         %# Change the axes tick marks
        'XTickLabel',{'Incompleted','Negative','Benign', 'Probably Benign', 'Suspicious Abnormality', 'Highly Suspicious Malignancy'},...  %#   and tick labels 'Incompleted', 'Normal', 'Benign', 'Malignant'
        'YTick',1:length_category,...
        'YTickLabel',{'Incompleted','Negative','Benign', 'Probably Benign', 'Suspicious Abnormality', 'Highly Suspicious Malignancy'},...
        'TickLength',[0 0]);
    
