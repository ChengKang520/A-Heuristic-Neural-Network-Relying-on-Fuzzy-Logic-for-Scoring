

clear all;
close all;
clc;

%% Firstly, set the parameter
pre_num = 1;
post_num = 1;
num_networks = 3;  % the number of networks
num_dataStrategy = 1; % the strategy of preprocessed data
num_bootstrap = 5; % the times of bootstrap
accuracy_matrix = zeros(3,6, num_bootstrap);
block_depth = 2;  %  the depth of residual blocks


%% load data

cdaddr=strcat('.\');
saveaddr='.\Buff_data\';
cd (cdaddr);


% save AUC_matrix_A_S2 accuracy_matrix

for num_block = 1:block_depth
    for data_Strategy = 2:3 %1:num_dataStrategy
        
        if data_Strategy == 3
            Whole = [];
            load('Data_3d_stacked_add_whiteedge.mat');
            Whole = Data_3d_stacked_addedge;
            if block_num_blockdepth == 1
                dataname = 'AddLayer_Combined_edge';
            else if num_block == 2
                    dataname = 'AddLayer2_Combined_edge';
                end
            end
        else     if data_Strategy == 2
                Whole = [];
                load('Data_3d_stacked.mat');
                Whole = Data_3d_stacked;
                if num_block == 1
                    dataname = 'AddLayer_Stacked_edge';
                else if num_block == 2
                        dataname = 'AddLayer2_Stacked_edge';
                    end
                end
            else     if data_Strategy == 1
                    Whole = [];
                    load('Data_3d.mat');
                    Whole = Data_3d;
                    if num_block == 1
                        dataname = 'AddLayer_Replicated';
                    else if num_block == 2
                            dataname = 'AddLayer2_Replicated';
                        end
                    end
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
            %         analyzeNetwork(net) % show the structure of ResNet18
            layers = net.Layers;
            
            % show the input size
            inputSize = net.Layers(1).InputSize
            numClasses = 4;%6/post_num; % the number of classes
            
            
            %%  1. transfer train; only train the last layer (fully connected layer)
            
            % change the connection tabel
            if length(net.Layers) == 347  % ResNet50
                
                if num_block == 1
                    % one residual block
                    layersTransfor = 'res5b';
                    layers = [
                        imageInputLayer([7 7 2048],'Name','data_input','Normalization','zerocenter')
                        %------------------ You have to check its length
                        net.Layers(333:344)
                        %------------------ You have to check its length
                        fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc4_softmax')
                        classificationLayer('Name','ClassificationLayer_fc4')
                        ];
                    lgraph = layerGraph(layers);
                    lgraph = connectLayers(lgraph,'res5b_relu','res5c/in2');
                else if num_block == 2
                        % two residual block
                        layersTransfor = 'res5c_relu';
                        layers = [
                            imageInputLayer([7 7 2048],'Name','data_input','Normalization','zerocenter')
                            %------------------ You have to check its length
                            net.Layers(321:344)
                            %------------------ You have to check its length
                            fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                            softmaxLayer('Name','fc4_softmax')
                            classificationLayer('Name','ClassificationLayer_fc4')
                            ];
                        lgraph = layerGraph(layers);
                        lgraph = removeLayers(lgraph,'res5a_branch1');
                        lgraph = removeLayers(lgraph,'bn5a_branch1');
                        lgraph = connectLayers(lgraph,'res5a_relu','res5b_branch2a');
                        
                        lgraph = connectLayers(lgraph,'res5a_relu','res5b/in2');
                        lgraph = connectLayers(lgraph,'res5b_relu','res5c/in2');
                        
                    end
                end
                
            else
                if  length(net.Layers) == 41  % VGG16
                    % change the structures of last 3 layers
                    layersTransfor = 'pool4';
                    layers = [
                        imageInputLayer([14 14 512],'Name','data','Normalization','zerocenter')
                        net.Layers(26:38)
                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc3_softmax')
                        classificationLayer('Name','ClassificationLayer_fc3')
                        ];
                    lgraph = layerGraph(layers);
                end
            end
            
            analyzeNetwork(lgraph);
            
            loaddata_trainFeatures = strcat(saveaddr,dataname,'_',NNname,'_trainFeatures.mat');
            loaddata_validationFeatures = strcat(saveaddr,dataname,'_',NNname,'_validationFeatures.mat');
            loaddata_testFeatures = strcat(saveaddr,dataname,'_',NNname,'_testFeatures.mat');
            
            %  load features
            trainFeatures = load(loaddata_trainFeatures);
            validationFeatures = load(loaddata_validationFeatures);
            testFeatures = load(loaddata_testFeatures);
            
            options = trainingOptions('adam',...
                'LearnRateSchedule','piecewise', ...
                'MiniBatchSize',12,...
                'MaxEpochs',18,...
                'LearnRateDropFactor',0.1, ...
                'LearnRateDropPeriod',13, ...
                'InitialLearnRate',0.001,...
                'SequencePaddingValue', 0,...
                'ValidationData',{validationFeatures.validationFeatures,validation_data.Labels}, ...
                'ValidationFrequency',30);
            
            
            % bootstrap processing  ---  60 times
            testLabels = test_data.Labels;
            for flag_bootstrap = 1:num_bootstrap
                clear netTransfer
                netTransfer = trainNetwork(trainFeatures.trainFeatures, feed_data.Labels', lgraph, options);
                %     netTransfer = trainNetwork_my(trainFeatures, feed_data.Labels', lgraph, options);
                
                predictedLabels = classify(netTransfer,testFeatures.testFeatures);
                accuracy_matrix(data_Strategy, network_num, flag_bootstrap) = sum(predictedLabels==testLabels)/numel(predictedLabels);
            end
            
            clear trainFeatures
            clear validationFeatures
            clear testFeatures
            clear net
            clear layers
            close all
        end
    end
    
    if num_block == 2
        save AUC_matrix_B2_block1 accuracy_matrix
    else     if data_Strategy == 1
            save AUC_matrix_B2_block2 accuracy_matrix
        end
    end

end


