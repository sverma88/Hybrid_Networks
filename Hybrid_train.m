function [V_TD, V_P, f_TD, f_P, BlkIdx, Tmean] = Hybrid_train(InImg,HDNet)
% =======INPUT=============
% InImg     Input images (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB)
% K         Number of Clusters
% HDNet    HDNet parameters (struct)
%       .HDNet.NumStages
%           the number of stages in HDNet; e.g., 2
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
% OutImg_final    lbp features (each row corresponds to lbp feature vector)
% V               learned Tensor filter banks (cell)
% =========================

addpath('./Utils')
addpath('./HDNet_func')

NumImg = length(InImg);
V_TD = cell(HDNet.NumStages,1);
V_P = cell(HDNet.NumStages,1);
Tmean = cell(HDNet.NumStages,1);
OutImg = InImg;
ImgIdx = (1:NumImg)';

clear InImg


%%%%%%%%%% Train TDNet and HDNet %%%%%%%%%%%%%%%%%

for stage = 1:HDNet.NumStages
    display(['Computing Tensor filter bank and its outputs at stage ' num2str(stage) '...'])
    
    tic
    Tensor_Patches = create_tensor_patches(OutImg, HDNet.PatchSize(stage));
    toc
    
    
    if(stage==1)
        filt = 3;
        [V_TD{stage}, Tmean{stage}] = TD_UTF(Tensor_Patches,[filt,filt,filt,filt],10^-6,10); %Compute Tensor Filter Banks
        V_P{stage} = PCA_FilterBank(OutImg, HDNet.PatchSize(stage), HDNet.NumFilters(stage)); % compute PCA filter banks
        
    else
        filt = 3;
        [V_TD{stage}, Tmean{stage}] = TD_UTF(Tensor_Patches,[filt,filt,filt],10^-6,10); %Compute Tensor Filter Banks
        V_P{stage} = PCA_FilterBank(OutImg, HDNet.PatchSize(stage), HDNet.NumFilters(stage)); % compute PCA filter banks
        
    end
    
    
    if stage ~= HDNet.NumStages % compute the PCA outputs only when it is NOT the last stage
        [OutImg_TD, ImgIdx_TD] = Tensor_output(OutImg, ImgIdx, ...
            HDNet.PatchSize(stage), V_TD{stage}, Tmean{stage});
        [OutImg_P, ImgIdx_P] = PCA_output(OutImg, ImgIdx, ...
            HDNet.PatchSize(stage), HDNet.NumFilters(stage), V_P{stage});
        
        OutImg = [OutImg_TD; OutImg_P];
        ImgIdx = [ImgIdx_P; ImgIdx_P];
    end
end



%%%%%%%% HDNet NonLinearity %%%%%%%%%

f_TD = cell(NumImg,1); % compute the HDNet training feature one by one
f_P = cell(NumImg,1);


for idx = 1:NumImg
    if 0==mod(idx,100); display(['Extracting HDNet feasture of the ' num2str(idx) 'th training sample...']); end
    OutImgIndex = ImgIdx_TD==idx; % select feature maps corresponding to image "idx" (outputs of the-last-but-one PCA filter bank)
    
    
    [OutImg_i, ImgIdx_i] = Tensor_output(OutImg_TD(OutImgIndex), ones(sum(OutImgIndex),1),...
        HDNet.PatchSize(end), V_TD{end},Tmean{end});  % compute the last PCA outputs of image "idx"
    
    [f_TD{idx}, BlkIdx_TD] = HashingHist(HDNet,ImgIdx_i,OutImg_i); % compute the feature of image "idx"
    
    OutImgIndex = ImgIdx_P==idx; % select feature maps corresponding to image "idx" (outputs of the-last-but-one PCA filter bank)
    
    [OutImg_i, ImgIdx_i] = PCA_output(OutImg_P(OutImgIndex), ones(sum(OutImgIndex),1),...
        HDNet.PatchSize(end), HDNet.NumFilters(end),V_P{end});  % compute the last PCA outputs of image "idx"
    
    [f_P{idx}, BlkIdx_P] = HashingHist(HDNet,ImgIdx_i,OutImg_i); % compute the feature of image "idx"
    
    %        [f{idx} BlkIdx] = SphereSum(HDNet,ImgIdx_i,OutImg_i); % Testing!!
    %     OutImg(OutImgIndex) = cell(sum(OutImgIndex),1);
    
    
    
end

f_TD = sparse([f_TD{:}]);
f_P = sparse([f_P{:}]);
BlkIdx = [BlkIdx_TD];


end







