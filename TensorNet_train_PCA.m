function [V, f, BlkIdx] = TensorNet_train_PCA(InImg,PCANet)
% =======INPUT=============
% InImg     Input images (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB) 
% K         Number of Clusters
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
% OutImg_final    lbp features (each row corresponds to lbp feature vector)
% V               learned Tensor filter banks (cell)
% =========================

addpath('./Utils')

NumImg = length(InImg);
V = cell(PCANet.NumStages,1); 
OutImg = InImg; 
ImgIdx = (1:NumImg)';
clear InImg

for stage = 1:PCANet.NumStages
    display(['Computing Tensor filter bank and its outputs at stage ' num2str(stage) '...'])
    
    tic
    Tensor_Patches = create_tensor_patches_revised(OutImg, PCANet.PatchSize(stage));
    toc
    
    
    if(stage==1)
        filt = PCANet.NumFilters(stage);
        V{stage} = TD_UTF(Tensor_Patches,[filt,filt,filt,filt],10^-6,10); %Compute Tensor Filter Banks
        
    else
        filt = 1+(PCANet.NumFilters(stage)^(1/3));
        V{stage} = TD_UTF(Tensor_Patches,[filt,filt,filt],10^-6,10); %Compute Tensor Filter Banks
    end
    

    if stage ~= PCANet.NumStages % compute the PCA outputs only when it is NOT the last stage
       [OutImg ImgIdx] = Tensor_output_revised(OutImg, ImgIdx, ...
            PCANet.PatchSize(stage), V{stage});       
    end
end


%%%%%%%% PCANet NonLinearity %%%%%%%%%

f = cell(NumImg,1); % compute the PCANet training feature one by one

for idx = 1:NumImg
    if 0==mod(idx,100); display(['Extracting PCANet feasture of the ' num2str(idx) 'th training sample...']); end
    OutImgIndex = ImgIdx==idx; % select feature maps corresponding to image "idx" (outputs of the-last-but-one PCA filter bank)
    
    
    [OutImg_i, ImgIdx_i] = Tensor_output_revised(OutImg(OutImgIndex), ones(sum(OutImgIndex),1),...
        PCANet.PatchSize(end), V{end});  % compute the last PCA outputs of image "idx"
    
    [f{idx}, BlkIdx] = HashingHist(PCANet,ImgIdx_i,OutImg_i); % compute the feature of image "idx"
    %        [f{idx} BlkIdx] = SphereSum(PCANet,ImgIdx_i,OutImg_i); % Testing!!
%     OutImg(OutImgIndex) = cell(sum(OutImgIndex),1);
    
end

f = sparse([f{:}]);
BlkIdx = [];

    
end







