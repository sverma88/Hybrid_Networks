function im_ten = im2tensor_mean_removal(varargin)
% 

NumInput = length(varargin);
InImg = varargin{1};
patchsize12 = varargin{2};
mean_removal= true;

z = size(InImg,3);
im = cell(z,1);
if NumInput == 2
    for i = 1:z
        iim = im2colstep(InImg(:,:,i),patchsize12);
        if(mean_removal)
            im{i} = bsxfun(@minus, iim, mean(iim))'; 
        else
            im{i} = iim';
        end
%         iim = bsxfun(@minus, iim, mean(iim)); 
%         im{i} = bsxfun(@minus, iim, mean(iim,2))';
    end
else
    for i = 1:z
        iim = im2colstep(InImg(:,:,i),patchsize12,varargin{3});
        if(mean_removal)
            im{i} = bsxfun(@minus, iim, mean(iim))'; 
        else
            im{i} = iim';
        end
%         iim = bsxfun(@minus, iim, mean(iim)); 
%         im{i} = bsxfun(@minus, iim, mean(iim,2))';
    end 
end
im = [im{:}]';

[~, num_patches] = size(im);

im_ten = tenzeros([num_patches,patchsize12(1), patchsize12(2)]);

for i=1:num_patches
    im_ten(i,:,:)=reshape(im(:,i),patchsize12(2), patchsize12(2));
end
    
    