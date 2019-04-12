% ==== HDNet Demo =======
% S. Verma, C. Wang, L. Zhu, W. Liu, 
% "Hybrid Networks: Improving Deep Learning Networks with two views of Images" ICONIP'18. 

% Sunny verma [Sunny.Verma@student.uts.edu.au]
% Please email me if you find bugs, or have suggestions or questions!
% ========================

parpool('local',20)
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
addpath('./TD_factorization');
addpath('./tensor_toolbox_2.6');
make;

ImgSize = 32; 

randfix = 5;
rng(randfix)

%%%% The amount of training data inetger between 500-5000
data_perc = 10; 

%% Loading data from CIFAR10 (50000 training, 10000 testing) 
DataPath = '/data/suverma/TKDE/TD_Capstone/cifar-10-batches-mat';

TrnLabels = [];
TrnData = [];
for i = 1:5
    load(fullfile(DataPath,['data_batch_' num2str(i) '.mat']));
    TrnData = [TrnData, data'];
    TrnLabels= [TrnLabels; labels];
end

load(fullfile(DataPath,'test_batch.mat'));
TestData = double(data)';
TestLabels = labels;

ImgFormat = 'color'; %'gray'

TrnLabels = double(TrnLabels);
TestLabels = double(TestLabels) + 1;


TrnData_aux = [];
TrnLabels_aux = [];

for class = 1:10
    rndindex=randperm(5000,data_perc);
    data_index = find(TrnLabels == class-1);
    data_index = data_index(rndindex);
    TrnData_aux = [TrnData_aux TrnData(:,data_index')];  % sample training samples
    TrnLabels_aux = [TrnLabels_aux; ones(data_perc,1)*class];
    
end

rndindex=randperm(numel(TrnLabels_aux));
TrnData_aux = TrnData_aux(:,rndindex');  % sample training samples
TrnLabels_aux = TrnLabels_aux(rndindex); % 
 
% %%%%%%%%%%%%%%%%%%%%%%%%
nTestImg = length(TestLabels);


%% HDNet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
HDNet.NumStages = 2;
HDNet.PatchSize = [5 5];
HDNet.NumFilters = [27 8];
HDNet.HistBlockSize = [8 8];
HDNet.BlkOverLapRatio = 0.5;
HDNet.Pyramid = [4 2 1];

fprintf('\n ====== HDNet Parameters ======= \n')
HDNet

fprintf('\n ====== RNG 5 ======= \n')


%% HDNet Training with 10000 samples
fprintf('\n ====== HDNet Training ======= \n')
TrnData_ImgCell = mat2imgcell(double(TrnData_aux),ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
TestData_ImgCell = mat2imgcell(double(TestData),ImgSize,ImgSize,ImgFormat);


tic
[V_TD, V_P, ftd, fp, BlkIdx, Tmean] = Hybrid_train(TrnData_ImgCell,HDNet);
toc


%% PCA hashing over histograms
c = 45;
    
fprintf('\n ====== Training Linear SVM Classifier ======= \n')
display(['now testing c = ' num2str(c) '...'])
models = train(TrnLabels_aux, [ftd; fp]', ['-s 1 -c ' num2str(c) ' -q']); % we use linear SVM classifier (C = 10), calling liblinear library

clear ftd;
clear fp;

fprintf('\n ====== HDNet Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

for idx = 1:1:nTestImg
    
    [ftest_TD,ftest_P] = Hybrid_FeaExt(TestData_ImgCell(idx),V_TD, V_P, HDNet, Tmean); % extract a test feature using trained HDNet model 

    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        [ftest_TD;ftest_P]', models, '-q');
    
    if xLabel_est == TestLabels(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    
    if 0==mod(idx,nTestImg/1000)
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]); 
    end 
    
end

Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy


%% Results display
fprintf('\n ===== Results of HDNet, followed by a linear SVM classifier =====');
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);


name = ['HDNet_',int2str(HDNet.NumFilters(1)),'_',int2str(HDNet.NumFilters(2)),'_',int2str(HDNet.PatchSize(1)),...
    '_',int2str(HDNet.PatchSize(2)),'_',int2str(HDNet.HistBlockSize(1)),'_',num2str(HDNet.BlkOverLapRatio),'.mat'];

save(name,'ErRate');



 