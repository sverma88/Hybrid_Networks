function [tensor_patches] = create_tensor_patches(InImg, PatchSize)

% =======INPUT=============
% InImg            Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% =======OUTPUT============
% tensor_patches   the patches in form of tensors
% =========================

addpath('./Utils')

% to efficiently cope with the large training samples, if the number of training we randomly subsample 10000 the
% training set to learn PCA filter banks
ImgZ = length(InImg);
MaxSamples = 150000;
NumRSamples = min(ImgZ, MaxSamples); 
RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);

InImg = InImg(RandIdx);
RandIdx = 1:NumRSamples;

%% Learning PCA filters (V)
NumChls = size(InImg{1},3);

tensor_patches = cell(NumRSamples,1);

if(NumChls >1)
    tic
    parfor i = 1:numel(RandIdx)
        im = im2col_mean_removal_tensor(InImg{RandIdx(i)},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal            
        tensor_patches{i} = im;
    end
       
    tensor_patches = tensor(cat(1,tensor_patches{:}));
    toc
else
    
    tic
    length(RandIdx)
    parfor i = 1:numel(RandIdx)
        im = im2col_mean_removal_tensor2d(InImg{RandIdx(i)},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal

        tensor_patches{i} = im;
    end
    
    tensor_patches = tensor(cat(1,tensor_patches{:}));
    toc
end

end




