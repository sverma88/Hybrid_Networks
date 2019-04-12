function im = im2col_mean_removal_tensor2d(varargin)
% 

NumInput = length(varargin);
InImg = varargin{1};
patchsize12 = varargin{2}; 

im = cell(1,1);
iim = im2colstep(InImg,patchsize12);
% im{1} = bsxfun(@minus, iim, mean(iim))'; 
im{1} =  iim;

% im = [im{:}]';
im = cell2mat(im);

im1 = arrayfun(@(n) conten_BW(im(:,n),patchsize12(1)), 1:size(im,2), 'UniformOutput', false);
im = permute(cat(3,im1{:}),[3,1,2]);
    
end

    
    