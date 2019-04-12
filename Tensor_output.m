function [OutImg OutImgIdx] = Tensor_output(InImg, InImgIdx, PatchSize, V, Tmean)
% Computing Tensor filter outputs
% ======== INPUT ============
% InImg         Input images (cell structure); each cell can be either a matrix (Gray) or a 3D tensor (RGB)   
% InImgIdx      Image index for InImg (column vector)
% PatchSize     Patch size (or filter size); the patch is set to be sqaure
% V             PCA filter banks (cell structure); V{i} for filter bank in the ith stage  
% ======== OUTPUT ===========
% OutImg           filter output (cell structure)
% OutImgIdx        Image index for OutImg (column vector)
% ===========================



addpath('./Utils')


[ImgX, ImgY, NumChls] = size(InImg{1});
if (NumChls >1)
    NumFilters = size(V{1,2},2)*size(V{1,3},2)*size(V{1,4},2);
    fu = size(V{1,2},2);
    fv = size(V{1,3},2);
    fw = size(V{1,4},2);

else
    NumFilters = size(V{1,2},2)^2;
end

Filters = cell(NumFilters,1);

filsize = size(V{1,2},2);


if(NumChls >1)
    cnt = 1;
    for i=1:fu
        for j=1:fv
            for k=1:fw
                Filters{cnt,1} = double(tenmat(tensor(permute(reshape(kron((V{1,2}(:,i)*V{1,3}(:,j)'),...
                    V{1,4}(:,k)),[3,PatchSize,PatchSize]),[2,3,1])),1:3,'t'));
                cnt = cnt+1;
            end
        end
    end

else
    cnt =1;
    for i=1:filsize
        for j=1:filsize
            Filters{cnt,1} = double(tenmat(tensor(V{1,2}(:,i)*V{1,3}(:,j)'),1:2,'t'));
            cnt = cnt+1;
        end
    end
    
    Filters{end-1} = Filters{end};
    NumFilters = NumFilters-1;

end        
     

ImgZ = length(InImg);
mag = (PatchSize-1)/2;
OutImg = cell(NumFilters*ImgZ,1); 
% cnt = 0;

for i = 1:ImgZ
    img = zeros(ImgX+PatchSize-1,ImgY+PatchSize-1, NumChls);
    img((mag+1):end-mag,(mag+1):end-mag,:) = InImg{i};     
    im = im2col_mean_removal_tenRec(img,[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    %%%% Subtract the mean  %%%%
    im = im - repmat(reshape(Tmean,numel(Tmean),1),1,size(im,2));
    
    aux = cell(NumFilters,1);
    
    parfor j=1:NumFilters
        aux{j} = reshape(Filters{j,1}*im,ImgX,ImgY);  % convolution output
    end
    
    cnt = 1 + (i-1)*NumFilters;
    endcnt = i*NumFilters;
    
    OutImg(cnt:endcnt,1) = aux;
%     for j = 1:NumFilters
%             cnt = cnt + 1;
%             OutImg{cnt} = reshape(Filters{j,1}*im,ImgX,ImgY);  % convolution output
%     end
    InImg{i} = [];
end

OutImgIdx = kron(InImgIdx,ones(NumFilters,1)); 

end






