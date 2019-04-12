function [ftd_pca, fp_pca] = Hybrid_FeaExt(InImg,V_TD,V_P,PCANet, Tmean)
% =======INPUT=============
% InImg     Input images (cell)  
% V         given PCA filter banks (cell)
% PCANet    PCANet parameters (struct)
%       .PCANet.NumStages      
%           the number of stages in PCANet; e.g., 2  
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., [5 3]
%           means patch size equalt to 5 and 3 in the first stage and second stage, respectively 
%       .NumFilters
%           the number of filters in each stage; e.g., [16 8] means 16 and
%           8 filters in the first stage and second stage, respectively
%       .HistBlockSize 
%           the size of each block for local histogram; e.g., [10 10]
%       .BlkOverLapRatio 
%           overlapped block region ratio; e.g., 0 means no overlapped 
%           between blocks, and 0.3 means 30% of blocksize is overlapped 
%       .Pyramid
%           spatial pyramid matching; e.g., [1 2 4], and [] if no Pyramid
%           is applied
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
% BlkIdx    index of local block from which the histogram is compuated
% =========================

addpath('./Utils')
addpath('./HDNet_func')


NumImg = length(InImg);

OutImg = InImg; 
ImgIdx = (1:NumImg)';

clear InImg;

stage=1;
[OutImg_TD, ImgIdx_TD] = Tensor_output(OutImg, ImgIdx, ...
    PCANet.PatchSize(stage), V_TD{stage}, Tmean{stage});

[OutImg_P, ImgIdx_P] = PCA_output(OutImg, ImgIdx, ...
    PCANet.PatchSize(stage),PCANet.NumFilters(stage), V_P{stage});


stage = 2;
[OutImg_TD, ImgIdx_TD] = Tensor_output(OutImg_TD, ImgIdx_TD, ...
    PCANet.PatchSize(stage),V_TD{stage}, Tmean{stage});


[OutImg_P, ImgIdx_P] = PCA_output(OutImg_P, ImgIdx_P, ...
    PCANet.PatchSize(stage),PCANet.NumFilters(stage), V_P{stage});


[f_TD,~] = HashingHist(PCANet,ImgIdx_TD,OutImg_TD);
[f_P,~] = HashingHist(PCANet,ImgIdx_P,OutImg_P);

ftd_pca = f_TD;
fp_pca = f_P;


end
%





