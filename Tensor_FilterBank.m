function [U,V] = Tensor_FilterBank(InImg, PatchSize, NumFilters) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of PCA filters in the bank.
% =======OUTPUT============
% V                PCA filter banks, arranged in column-by-column manner
% =========================

addpath('./Utils')

% to efficiently cope with the large training samples, if the number of training we randomly subsample 10000 the
% training set to learn PCA filter banks
ImgZ = length(InImg);
MaxSamples = 100000;
NumRSamples = min(ImgZ, MaxSamples); 
RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);
[num_rows, num_cols] = size(InImg{1});
num_neighbor = (num_rows-PatchSize +1)*(num_cols - PatchSize+1);

%% Learning PCA filters (V)
NumChls = size(InImg{1},3);
% Rx = zeros(NumChls*PatchSize^2,NumChls*PatchSize^2);
Rx = tenzeros([ImgZ, num_neighbor, PatchSize, PatchSize]);

for i = RandIdx %1:ImgZ
%     i
%     im = im2col_mean_removal(InImg{i},[PatchSize PatchSize]);
    im = im2tensor_mean_removal(InImg{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    Rx(i,:,:,:) = im; % sum of all the input images' covariance matrix
end

u= TD_UTF(Rx,[3,3,3],10^-6,300);
% Rx = Rx/(NumRSamples*size(im,2));
% [E D] = eig(Rx); 
% [~, ind] = sort(diag(D),'descend');
% V = E(:,ind(1:NumFilters));  % principal eigenvectors 



 



