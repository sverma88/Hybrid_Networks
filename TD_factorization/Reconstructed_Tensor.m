% Function to Reconstruct Tensor, given its factors Matrices and core
% tensor

function [Reconstructed_Tensor]=Reconstructed_Tensor(Core_Tensors,Singular_Factors)

%Input
% Core_Tensor               : Low Rank Core Tensor  
% Singular_Factors          : Cell contains Low Rank Singular Factors 
%
% Output
% Reconstructed_Tensor      : Reconstructed Tensor using Core tensor and Singular Factors 
%
% Author                    : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update               : 05/04/2018

% %
% Mode-1 = #Images, Mode-2 = #patches 

Number_Modes=ndims(Core_Tensors);
   

for i=1:Number_Modes
    Mode_to_Operate=i;
    Core_Tensors= ttm(Core_Tensors,Singular_Factors{1,i},Mode_to_Operate);
    
end

Reconstructed_Tensor=Core_Tensors;
    
end

