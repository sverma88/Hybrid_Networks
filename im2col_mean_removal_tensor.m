function im = im2col_mean_removal_tensor(varargin)
% 

NumInput = length(varargin);
InImg = varargin{1};
patchsize12 = varargin{2}; 

z = size(InImg,3);
im = cell(z,1);
if NumInput == 2
    for i = 1:z
        iim = im2colstep(InImg(:,:,i),patchsize12);        
        im{i} = iim; 
%         im{i} = bsxfun(@minus, iim, mean(iim))'; 
%         iim = bsxfun(@minus, iim, mean(iim)); 
%         im{i} = bsxfun(@minus, iim, mean(iim,2))';
    end
else
    for i = 1:z
        iim = im2colstep(InImg(:,:,i),patchsize12,varargin{3});
        im{i} = iim;
%         im{i} = bsxfun(@minus, iim, mean(iim))'; 
%         iim = bsxfun(@minus, iim, mean(iim)); 
%         im{i} = bsxfun(@minus, iim, mean(iim,2))';
    end 
end
% im = [im{:}]';
im = cell2mat(im);

if(z>1)    
    im1 = arrayfun(@(n) conten(im(:,n),patchsize12(1)), 1:size(im,2), 'UniformOutput', false);
    im = permute(cat(4,im1{:}),[4,1,2,3]);
else
    im1 = arrayfun(@(n) conten_BW(im(:,n),patchsize12(1)), 1:size(im,2), 'UniformOutput', false);
    im = permute(cat(3,im1{:}),[3,1,2]);
    
end

    
    