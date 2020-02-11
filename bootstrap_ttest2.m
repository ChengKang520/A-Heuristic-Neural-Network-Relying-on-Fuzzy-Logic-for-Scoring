function [Ttest] = bootstrap_ttest2(X,Y,BTN,C_level)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%         X and Y are the sample
%         BTN is the times of bootstrap
%         C_level is the conficence interval
% Output
%         Ttest is the P and T value
%         CI is the value in the 95% and 5% position
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

XX = zeros(1,BTN);
YY = zeros(1,BTN);

XX = X(randi([1,length(X)], 1,BTN));
YY = Y(randi([1,length(X)], 1,BTN));
XX = sort(XX);
YY = sort(YY);

[h,p,ci,stats]=ttest2(XX,YY,C_level,0);
          
Ttest = [p, stats.tstat];

end



