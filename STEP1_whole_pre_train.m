
clear all;
close all;
clc;

%% load data

load('Data_6_Class.mat');

% using diferent operator to process the raw image, finally, 
%   the canny operator performed best
Mass_Train_Input = Data.Mass.Train.Input;
Mass_Test_Input = Data.Mass.Test.Input;
Calc_Train_Input = Data.Calc.Train.Input;
Calc_Test_Input = Data.Calc.Test.Input;

%% Stack the data with raw image, processed result of two edge extrcation operators
% constract the new data -- 224*224 (raw); 224*224 (raw)
Data_3d = Data;

Data_3d.Mass.Train.Input(:,:,1,:) = Mass_Train_Input;
Data_3d.Mass.Train.Input(:,:,2,:) = Mass_Train_Input;
Data_3d.Mass.Train.Input(:,:,3,:) = Mass_Train_Input;

Data_3d.Mass.Test.Input(:,:,1,:) = Mass_Test_Input;
Data_3d.Mass.Test.Input(:,:,2,:) = Mass_Test_Input;
Data_3d.Mass.Test.Input(:,:,3,:) = Mass_Test_Input;

Data_3d.Calc.Train.Input(:,:,1,:) = Calc_Train_Input;
Data_3d.Calc.Train.Input(:,:,2,:) = Calc_Train_Input;
Data_3d.Calc.Train.Input(:,:,3,:) = Calc_Train_Input;

Data_3d.Calc.Test.Input(:,:,1,:) = Calc_Test_Input;
Data_3d.Calc.Test.Input(:,:,2,:) = Calc_Test_Input;
Data_3d.Calc.Test.Input(:,:,3,:) = Calc_Test_Input;

Data_3d_stacked = Data_3d;
Data_3d_stacked_addedge = Data_3d;

% constract the new data -- 224*224 (canny operator); 224*224 (log operator)
%  Mass Train dataset
Samples_input_number = [];
Samples_input_number = size(Mass_Train_Input);
for i = 1:Samples_input_number(4)
    Sample_input = [];
    Input_Log_Operator = [];
    Input_Canny_Operator = [];
    Sample_input_ED4 = [];
    Sample_input_ED5 = [];
    
    Sample_input = Mass_Train_Input(:,:,1,i);
    Input_Log_Operator = edge(Sample_input,'log');
    Input_Canny_Operator = edge(Sample_input,'canny');
    
    Data_3d_stacked.Mass.Train.Input(:,:,2,i) = uint8(Input_Log_Operator)*255;
    Data_3d_stacked.Mass.Train.Input(:,:,3,i) = uint8(Input_Canny_Operator)*255;
    
    %  adding two different edge features in raw image
    Sample_input_ED4 = Sample_input;
    Sample_input_ED4(find(Input_Log_Operator == 1)) = 255;
    Sample_input_ED5 = Sample_input;
    Sample_input_ED5(find(Input_Canny_Operator == 1)) = 255;
    Data_3d_stacked_addedge.Mass.Train.Input(:,:,2,i) = Sample_input_ED4;
    Data_3d_stacked_addedge.Mass.Train.Input(:,:,3,i) = Sample_input_ED5;
% figure;
% subplot(2,2,1);imshow(Data_3d_stacked.Mass.Train.Input(:,:,2,i));
% subplot(2,2,2);imshow(Data_3d_stacked.Mass.Train.Input(:,:,3,i));
% subplot(2,2,3);imshow(Data_3d_stacked_addedge.Mass.Train.Input(:,:,2,i));
% subplot(2,2,4);imshow(Data_3d_stacked_addedge.Mass.Train.Input(:,:,3,i));
end

%  Mass Test dataset
Samples_input_number = [];
Samples_input_number = size(Mass_Test_Input);
for i = 1:Samples_input_number(4)
    Sample_input = [];
    Input_Log_Operator = [];
    Input_Canny_Operator = [];
    Sample_input_ED4 = [];
    Sample_input_ED5 = [];
    
    Sample_input = Mass_Test_Input(:,:,1,i);
    Input_Log_Operator = edge(Sample_input,'log');
    Input_Canny_Operator = edge(Sample_input,'canny');
    
    Data_3d_stacked.Mass.Test.Input(:,:,2,i) = uint8(Input_Log_Operator)*255;
    Data_3d_stacked.Mass.Test.Input(:,:,3,i) = uint8(Input_Canny_Operator)*255;
    
    %  adding two different edge features in raw image
    Sample_input_ED4 = Sample_input;
    Sample_input_ED4(find(Input_Log_Operator == 1)) = 255;
    Sample_input_ED5 = Sample_input;
    Sample_input_ED5(find(Input_Canny_Operator == 1)) = 255;
    Data_3d_stacked_addedge.Mass.Test.Input(:,:,2,i) = Sample_input_ED4;
    Data_3d_stacked_addedge.Mass.Test.Input(:,:,3,i) = Sample_input_ED5;
% figure;
% subplot(2,2,1);imshow(Data_3d_stacked.Mass.Test.Input(:,:,2,i));
% subplot(2,2,2);imshow(Data_3d_stacked.Mass.Test.Input(:,:,3,i));
% subplot(2,2,3);imshow(Data_3d_stacked_addedge.Mass.Test.Input(:,:,2,i));
% subplot(2,2,4);imshow(Data_3d_stacked_addedge.Mass.Test.Input(:,:,3,i));
end

%  Calc Train dataset
Samples_input_number = [];
Samples_input_number = size(Calc_Train_Input);
for i = 1:Samples_input_number(4)
    Sample_input = [];
    Input_Log_Operator = [];
    Input_Canny_Operator = [];
    Sample_input_ED4 = [];
    Sample_input_ED5 = [];
    
    Sample_input = Calc_Train_Input(:,:,1,i);
    Input_Log_Operator = edge(Sample_input,'log');
    Input_Canny_Operator = edge(Sample_input,'canny');
    
    Data_3d_stacked.Calc.Train.Input(:,:,2,i) = uint8(Input_Log_Operator)*255;
    Data_3d_stacked.Calc.Train.Input(:,:,3,i) = uint8(Input_Canny_Operator)*255;
    
    %  adding two different edge features in raw image
    Sample_input_ED4 = Sample_input;
    Sample_input_ED4(find(Input_Log_Operator == 1)) = 255;
    Sample_input_ED5 = Sample_input;
    Sample_input_ED5(find(Input_Canny_Operator == 1)) = 255;
    Data_3d_stacked_addedge.Calc.Train.Input(:,:,2,i) = Sample_input_ED4;
    Data_3d_stacked_addedge.Calc.Train.Input(:,:,3,i) = Sample_input_ED5;
% figure;
% subplot(2,2,1);imshow(Data_3d_stacked.Calc.Train.Input(:,:,2,i));
% subplot(2,2,2);imshow(Data_3d_stacked.Calc.Train.Input(:,:,3,i));
% subplot(2,2,3);imshow(Data_3d_stacked_addedge.Calc.Train.Input(:,:,2,i));
% subplot(2,2,4);imshow(Data_3d_stacked_addedge.Calc.Train.Input(:,:,3,i));
end

%  Calc Test dataset
Samples_input_number = [];
Samples_input_number = size(Calc_Test_Input);
for i = 1:Samples_input_number(4)
    Sample_input = [];
    Input_Log_Operator = [];
    Input_Canny_Operator = [];
    Sample_input_ED4 = [];
    Sample_input_ED5 = [];
    
    Sample_input = Calc_Test_Input(:,:,1,i);
    Input_Log_Operator = edge(Sample_input,'log');
    Input_Canny_Operator = edge(Sample_input,'canny');
    
    Data_3d_stacked.Calc.Test.Input(:,:,2,i) = uint8(Input_Log_Operator)*255;
    Data_3d_stacked.Calc.Test.Input(:,:,3,i) = uint8(Input_Canny_Operator)*255;
    
    %  adding two different edge features in raw image
    Sample_input_ED4 = Sample_input;
    Sample_input_ED4(find(Input_Log_Operator == 1)) = 255;
    Sample_input_ED5 = Sample_input;
    Sample_input_ED5(find(Input_Canny_Operator == 1)) = 255;
    Data_3d_stacked_addedge.Calc.Test.Input(:,:,2,i) = Sample_input_ED4;
    Data_3d_stacked_addedge.Calc.Test.Input(:,:,3,i) = Sample_input_ED5;
% figure;
% subplot(2,2,1);imshow(Data_3d_stacked.Calc.Test.Input(:,:,2,i));
% subplot(2,2,2);imshow(Data_3d_stacked.Calc.Test.Input(:,:,3,i));
% subplot(2,2,3);imshow(Data_3d_stacked_addedge.Calc.Test.Input(:,:,2,i));
% subplot(2,2,4);imshow(Data_3d_stacked_addedge.Calc.Test.Input(:,:,3,i));
end

% save Data_3d Data_3d
% save Data_3d_stacked Data_3d_stacked
% save Data_3d_stacked_addedge Data_3d_stacked_addedge
% save Data_3d_stacked_add_whiteedge Data_3d_stacked_addedge
