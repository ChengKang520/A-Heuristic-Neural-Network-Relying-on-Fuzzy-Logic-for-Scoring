
clear all;
clc;
load('AUC_matrix_C.mat');
num_NN = size(accuracy_matrix,1)
BTN = 200;  %  the times of bootstrap
C_level = 0.05;  % the confidence interval
Stat_value = zeros(2,num_NN,num_NN); % P_value*T_value
CI_value = zeros(3,num_NN); % 5% and 95% confidence interval, and mean value
for matrix_ttest1 = 1:num_NN
    
    sample_A = [];
    sample_A = squeeze(accuracy_matrix(matrix_ttest1, :));
    % compute the confidence interval 
    XX = sample_A(randi([1,length(sample_A)], 1,BTN));
    XX = sort(XX);
    CI_value(:,matrix_ttest1) = [XX(BTN*0.05); XX(BTN*0.95); mean(XX(BTN*0.05:BTN*0.95))];

    % T-test
    for matrix_ttest2 = 1:num_NN
        sample_B = [];
        sample_B = squeeze(accuracy_matrix(matrix_ttest2, :));
        [Stat_value(:, matrix_ttest1, matrix_ttest2)] = bootstrap_ttest2(sample_A, sample_B, BTN, C_level);
    end
end

P_value = squeeze(Stat_value(1,:,:));
T_value = squeeze(Stat_value(2,:,:));

P_value(find(P_value < 0.01)) = 0;
P_value(find(P_value < 0.01)) = 0;
