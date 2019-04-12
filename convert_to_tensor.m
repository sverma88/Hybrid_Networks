function [tensor_patches] = convert_to_tensor(patch, PatchSize, NumChnls)
% =======INPUT=============
% patch            Image patches   
% shape            Shape of the patch 
% NumChnls         Number of Channels
% =======OUTPUT============
% tensor_patches   Patches converted into tensor 
% =========================

K = size(patch,2);

if (NumChnls >1)
    
    tensor_patches = zeros(K,PatchSize,PatchSize,3);
    parfor j=1:K
        A = zeros(PatchSize, PatchSize, 3);
        A(:,:,1)=reshape(patch(1:PatchSize^2,j),PatchSize,PatchSize);
        A(:,:,2)=reshape(patch(PatchSize^2+1:2*PatchSize^2,j),PatchSize,PatchSize);
        A(:,:,3)=reshape(patch(2*PatchSize^2+1:end,j),PatchSize,PatchSize);
        tensor_patches(j, : ,:,:) = A;
    end

else
    
    tensor_patches = zeros(K,PatchSize,PatchSize);
    parfor j=1:K
        tensor_patches(j,:,:)=reshape(patch(:,j),PatchSize,PatchSize);
    end

    
end