
clear all;
close all;
clc;

pre_num = 1;
post_num = 1;
num_networks = 6;  % the number of networks
num_dataStrategy = 1; % the strategy of preprocessed data
%% load data

cdaddr=strcat('.\');
saveaddr='.\Buff_data\';
cd (cdaddr);

for data_Strategy = 1 %1:num_dataStrategy
    
    if data_Strategy == 3
        Whole = [];
        load('Data_3d_stacked_add_whiteedge.mat');
        Whole = Data_3d_stacked_addedge;
        dataname = 'Combined_edge';
    else     if data_Strategy == 2
            Whole = [];
            load('Data_3d_stacked.mat');
            Whole = Data_3d_stacked;
            dataname = 'Stacked_edge';
        else     if data_Strategy == 1
                Whole = [];
                load('Data_3d.mat');
                Whole = Data_3d;
                dataname = 'Replicated';
            end
        end
    end
    
    % check the image
    subject_num = 2
    subplot(1,3,1);imshow(Whole.Mass.Train.Input(:,:,1,subject_num));
    subplot(1,3,2);imshow(Whole.Mass.Train.Input(:,:,2,subject_num));
    subplot(1,3,3);imshow(Whole.Mass.Train.Input(:,:,3,subject_num));
    
    %% the loop for traning task -- 5 different NN
    for network_num = 3 %1:num_networks
        
        if network_num == 1
            net = resnet18;
            NNname = 'Resnet18';
        else     if network_num == 2
                net = resnet50;
                NNname = 'Resnet50';
            else     if network_num == 3
                    net = resnet101;
                    NNname = 'Resnet101';
                else     if network_num == 4
                        net = vgg16;
                        NNname = 'VGG16';
                    else     if network_num == 5
                            net = vgg19;
                            NNname = 'VGG19';
                        else if network_num == 6
                                net = googlenet;
                                NNname = 'GoogleNet';
%                             else if network_num == 7
%                                     net = inceptionv3;
%                                     NNname = 'InceptionV3';
%                                 end
                            end
                        end
                    end
                end
            end
        end
        
        
        %% Load Data -- Feeding the data set into the network
        %  ---------------------------------------------------------------------------------------------------------
        
        %%  Only Mass Dataset -- 2 or 3-class classification
        %     feed_data.Images = Whole.Mass.Train.Input;
        %     feed_data.Labels = categorical((Whole.Mass.Train.Target + pre_num) / post_num);
        %
        %     validation_data.Images = feed_data.Images(:,:,:,1:131);
        %     validation_data.Labels = feed_data.Labels(1:131);
        %
        %     test_data.Images = Whole.Mass.Test.Input;
        %     test_data.Labels = categorical((Whole.Mass.Test.Target + pre_num) / post_num);
        
        %%  Mass and Calc Dataset
        
        feed_data.Images = cat(4, Whole.Mass.Train.Input, Whole.Calc.Train.Input);
        validation_data.Images = feed_data.Images(:,:,:,1:287);
        test_data.Images = cat(4, Whole.Mass.Test.Input, Whole.Calc.Test.Input);
        
        %  Mass and Calc Dataset -- 4-class classification
        feed_data.Labels = [Whole.Mass.Train.Target; Whole.Calc.Train.Target];
        test_data.Labels = [Whole.Mass.Test.Target; Whole.Calc.Test.Target];
        
        % the distribution of samples
%         figure;
%         C = categorical(feed_data.Labels,[0 1 2 3 4 5],{'Incompleted','Negative','Benign', 'Probably Benign', 'Suspicious Abnormality', 'Highly Suspicious Malignancy'});
%         myhist(C);grid on;
        
        %  Mass and Calc Dataset -- 4-class classification
        feed_data.Labels = (feed_data.Labels + pre_num) / post_num;
        feed_data.Labels(find(feed_data.Labels == 4)) = 3;
        feed_data.Labels(find(feed_data.Labels >= 5)) = 4;
        feed_data.Labels = categorical(feed_data.Labels);
        
        validation_data.Labels = feed_data.Labels(1:287);
        
        test_data.Labels = (test_data.Labels + pre_num) / post_num;
        test_data.Labels(find(test_data.Labels == 4)) = 3;
        test_data.Labels(find(test_data.Labels >= 5)) = 4;
        test_data.Labels = categorical(test_data.Labels);
        
        
        %% Load ResNet* Module
        %     analyzeNetwork(net) % show the structure of ResNet18
        layers = net.Layers;
        
        % show the input size
        inputSize = net.Layers(1).InputSize
        numClasses = 4;%6/post_num; % the number of classes
        
        
        %%  1. transfer train; only train the last layer (fully connected layer)
        
        % change the connection tabel
        if length(net.Layers) == 72  % ResNet18
            % change the structures of last 3 layers
            layersTransfor = 'pool5';
            layers = [
                imageInputLayer([1 1 512],'Name','data','Normalization','zerocenter')
                fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                softmaxLayer('Name','fc4_softmax')
                classificationLayer('Name','ClassificationLayer_fc4')
                ];
            
        else if length(net.Layers) == 177  % ResNet50
                % change the structures of last 3 layers
                layersTransfor = 'avg_pool';
                layers = [
                    imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
                    fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                    softmaxLayer('Name','fc4_softmax')
                    classificationLayer('Name','ClassificationLayer_fc4')
                    ];
                
            else if length(net.Layers) == 347  % ResNet101
                    % change the structures of last 3 layers
                    layersTransfor = 'pool5';
                    layers = [
                        imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
                        fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc4_softmax')
                        classificationLayer('Name','ClassificationLayer_fc4')
                        ];
                    
                else if length(net.Layers) == 25  % AlexNet
                        % change the structures of last 3 layers
                        layersTransfor = 'drop7';
                        layers = [
                            imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
                            fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                            softmaxLayer('Name','fc4_softmax')
                            classificationLayer('Name','ClassificationLayer_fc4')
                            ];
                    else if length(net.Layers) == 41  % VGG16
                            % change the structures of last 3 layers
                            layersTransfor = 'drop7';
                            layers = [
                                imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
                                fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                                softmaxLayer('Name','fc4_softmax')
                                classificationLayer('Name','ClassificationLayer_fc4')
                                ];
                        else if length(net.Layers) == 47  % VGG19
                                % change the structures of last 3 layers
                                layersTransfor = 'drop7';
                                layers = [
                                    imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
                                    fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                                    softmaxLayer('Name','fc4_softmax')
                                    classificationLayer('Name','ClassificationLayer_fc4')
                                    ];
                                
                            else if length(net.Layers) == 144  % GoogleNet
                                    % change the structures of last 3 layers
                                    layersTransfor = 'pool5-drop_7x7_s1';
                                    layers = [
                                        imageInputLayer([1 1 1024],'Name','data','Normalization','zerocenter')
                                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                                        softmaxLayer('Name','fc3_softmax')
                                        classificationLayer('Name','ClassificationLayer_fc3')
                                        ];
%                                 else if length(net.Layers) == 316  % InceptionV3
%                                         % change the structures of last 3 layers
%                                         layersTransfor = 'avg_pool';
%                                         layers = [
%                                             imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
%                                             fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%                                             softmaxLayer('Name','fc3_softmax')
%                                             classificationLayer('Name','ClassificationLayer_fc3')
%                                             ];
%                                     end
                                end
                            end
                        end
                    end
                end
            end
        end
        
        lgraph = layerGraph(layers);
        
        % ---------------------augmentation operation
        imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandXTranslation',[-30,30],'RandYTranslation',[-30,30]);
        augimdsTrain = augmentedImageDatastore(inputSize(1:2), feed_data.Images, 'DataAugmentation', imageAugmenter);
        augimdsValidation =augmentedImageSource(inputSize(1:2), validation_data.Images, 'DataAugmentation', imageAugmenter );

%         augimdsTrain = feed_data.Images;
%         augimdsValidation = validation_data.Images;
        augimdsTest = test_data.Images;
        % extracting some specific layers, and ploting
        trainFeatures = activations(net, augimdsTrain, layersTransfor);
        validationFeatures = activations(net, augimdsValidation, layersTransfor);
        testFeatures = activations(net, augimdsTest, layersTransfor);
        
        options = trainingOptions('adam',...
            'LearnRateSchedule','piecewise', ...
            'MiniBatchSize',16,...
            'MaxEpochs',30,...
            'LearnRateDropFactor',0.1, ...
            'LearnRateDropPeriod',13, ...
            'InitialLearnRate',0.001,...
            'SequencePaddingValue', 0,...
            'ValidationData',{validationFeatures,validation_data.Labels}, ...
            'ValidationFrequency',30,...
            'Plots','training-progress');
        
        netTransfer = trainNetwork(trainFeatures, feed_data.Labels, lgraph, options);
        %     netTransfer = trainNetwork_my(trainFeatures, feed_data.Labels', lgraph, options);
        
        predictedLabels = classify(netTransfer,testFeatures);
        testLabels = test_data.Labels;
        
        accuracy = sum(predictedLabels==testLabels)/numel(predictedLabels)
        
        [predictedLabels scoresToReturn] = classify(netTransfer,testFeatures);
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
        text(0.5, 0.4, 'Quadruple Classification', 'fontsize',14);
        text(0.5, 0.35, strcat('AUC:',num2str(floor(accuracy*10000)/100), '%'),'fontsize',12,'color',[1 0 0]);
        
        savedata_name = strcat(dataname,'_',NNname,'_netTransfer');
        savedata_netTransfer = strcat(dataname,'_',NNname,'_netTransfer.mat');
        savedata_trainFeatures = strcat(dataname,'_',NNname,'_trainFeatures.mat');
        savedata_validationFeatures = strcat(dataname,'_',NNname,'_validationFeatures.mat');
        savedata_testFeatures = strcat(dataname,'_',NNname,'_testFeatures.mat');
        
        save([saveaddr, savedata_netTransfer], 'netTransfer');
        save([saveaddr, savedata_trainFeatures], 'trainFeatures');
        save([saveaddr, savedata_validationFeatures], 'validationFeatures');
        save([saveaddr, savedata_testFeatures], 'testFeatures');
        saveas(gcf,[saveaddr, savedata_name, '.bmp']);
        

        figure;
        confusion_matrix(testLabels,predictedLabels);
        saveas(gcf,[saveaddr, savedata_name, '_ConfusionMatrix.bmp']);
        
        
        clear netTransfer
        clear augimdsTrain
        clear augimdsValidation
        clear augimdsTest
        clear trainFeatures
        clear validationFeatures
        clear testFeatures
        clear net
        clear layers
        close all
    end
end







