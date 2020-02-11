
clear all;
close all;
clc;


%% load data
num_bootstrap = 10; % the times of bootstrap
accuracy_matrix = zeros(3,6, num_bootstrap);

cdaddr=strcat('.\');
saveaddr='.\Buff_data\';
cd (cdaddr);

Whole = [];
load('Data_3d.mat');
Whole = Data_3d;


% check the image
subject_num = 2
subplot(1,3,1);imshow(Whole.Mass.Train.Input(:,:,1,subject_num));
subplot(1,3,2);imshow(Whole.Mass.Train.Input(:,:,2,subject_num));
subplot(1,3,3);imshow(Whole.Mass.Train.Input(:,:,3,subject_num));

%% the loop for traning task -- 5 different NN
net = resnet101;
NNname = '6C_Resnet101';

%  ---------------------------------------------------------------------------------------------------------
%%  Mass and Calc Dataset

feed_data.Images = cat(4, Whole.Mass.Train.Input, Whole.Calc.Train.Input);
validation_data.Images = feed_data.Images(:,:,:,1:287);
test_data.Images = cat(4, Whole.Mass.Test.Input, Whole.Calc.Test.Input);

%  Mass and Calc Dataset -- 4-class classification
feed_data.Labels = [Whole.Mass.Train.Target; Whole.Calc.Train.Target];
test_data.Labels = [Whole.Mass.Test.Target; Whole.Calc.Test.Target];

%  Mass and Calc Dataset -- 4-class classification
feed_data.Labels = (feed_data.Labels + 1);
feed_data.Labels = feed_data.Labels;

validation_data.Labels = feed_data.Labels(1:287);

test_data.Labels = (test_data.Labels + 1);
test_data.Labels = test_data.Labels;

save feed_data feed_data
save validation_data validation_data
save test_data test_data

%% Load ResNet* Module
% analyzeNetwork(net) % show the structure of ResNet18
layers = net.Layers;

% show the input size
inputSize = net.Layers(1).InputSize
numClasses = 6; % the number of classes


%%  1. transfer train; only train the last layer (fully connected layer)

% change the connection tabel

layersTransfor = 'pool5';
layers = [
    imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
    fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
    softmaxLayer('Name','fc4_softmax')
    classificationLayer('Name','ClassificationLayer_fc4')
    ];

lgraph = layerGraph(layers);

loaddata_trainFeatures = strcat(saveaddr,dataname,'_',NNname,'_trainFeatures.mat');
loaddata_validationFeatures = strcat(saveaddr,dataname,'_',NNname,'_validationFeatures.mat');
loaddata_testFeatures = strcat(saveaddr,dataname,'_',NNname,'_testFeatures.mat');

%  load features
trainFeatures = load(loaddata_trainFeatures);
validationFeatures = load(loaddata_validationFeatures);
testFeatures = load(loaddata_testFeatures);

options = trainingOptions('adam',...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',16,...
    'MaxEpochs',30,...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',13, ...
    'InitialLearnRate',0.001,...
    'SequencePaddingValue', 0,...
    'ValidationData',{validationFeatures.validationFeatures,validation_data.Labels}, ...
    'ValidationFrequency',30,...
    'Plots','training-progress');

% train network 
netTransfer = trainNetwork_my(trainFeatures.trainFeatures, feed_data.Labels, lgraph, options);

% calculate the AUC
[predictedLabels scoresToReturn] = classify(netTransfer,testFeatures.testFeatures);
testLabels = test_data.Labels;
accuracy = sum(predictedLabels==testLabels)/numel(predictedLabels)

%% plot ROC curve
testLabels_ROC = zeros(numClasses,length(testLabels));
[GN_test, ~, G_testLabels] = unique(testLabels);

for j = 1:numClasses
    testLabels_ROC(j, find(G_testLabels == j)) = 1;
end

figure;
plotroc_my(testLabels_ROC,scoresToReturn', accuracy);
text(0.5, 0.4, 'Six Classification', 'fontsize',14);
text(0.5, 0.35, strcat('AUC:',num2str(floor(accuracy*10000)/100), '%'),'fontsize',12,'color',[1 0 0]);

savedata_name = strcat(dataname,'_',NNname,'_netTransfer_Fuzzy');
savedata_netTransfer = strcat(dataname,'_',NNname,'_netTransfer_Fuzzy.mat');
savedata_trainFeatures = strcat(dataname,'_',NNname,'_trainFeatures_Fuzzy.mat');
savedata_validationFeatures = strcat(dataname,'_',NNname,'_validationFeatures_Fuzzy.mat');
savedata_testFeatures = strcat(dataname,'_',NNname,'_testFeatures_Fuzzy.mat');

save([saveaddr, savedata_netTransfer], 'netTransfer');
save([saveaddr, savedata_trainFeatures], 'trainFeatures');
save([saveaddr, savedata_validationFeatures], 'validationFeatures');
save([saveaddr, savedata_testFeatures], 'testFeatures');
saveas(gcf,[saveaddr, savedata_name, '.bmp']);

figure;
confusion_matrix(testLabels,predictedLabels);
% saveas(gcf,[saveaddr, savedata_name, '_ConfusionMatrix.bmp']);

% set the label
xtb = get(gca,'XTickLabel');
ytb = get(gca,'YTickLabel');
xt = get(gca,'XTick');
yt = get(gca,'YTick');      

xtextp_x=xt;                     
ytextp_x=(yt(end)+0.5)*ones(1,length(xt));

xtextp_y=(xt(1)-0.5)*ones(1,length(yt));                     
ytextp_y=yt;

text(xtextp_x,ytextp_x,xtb,'HorizontalAlignment','right','rotation',45,'fontsize',10); 
text(xtextp_y,ytextp_y,ytb,'HorizontalAlignment','right','rotation',45,'fontsize',10); 
set(gca, 'Position', [0.13,0.390476190476191,0.795,0.534523809523811], 'OuterPosition', [0.053354838709677,0.318331872626351,1.025806451612903,0.655857434998541])
set(gca,'xticklabel','');% 将原有的标签隐去
set(gca,'yticklabel','');% 将原有的标签隐去

% %% bootstrap processing  ---  60 times
% testLabels = test_data.Labels;
% for flag_bootstrap = 1:num_bootstrap
%     clear netTransfer
%     netTransfer = trainNetwork_my(trainFeatures.trainFeatures, feed_data.Labels', lgraph, options);
%     
%     predictedLabels = classify(netTransfer,testFeatures.testFeatures);
%     accuracy_matrix(1, 1, flag_bootstrap) = sum(predictedLabels==testLabels)/numel(predictedLabels);
% end
% 
% save AUC_matrix_C_S1_Fuzzy_6c accuracy_matrix

