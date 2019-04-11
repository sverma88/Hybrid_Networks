% Function to find core tensor for category X tensor utilizing Factors Matrices 
% from the training phase of the Tensor itself

function [Core_Tensor]=Core_Tensor(Train_Tensors,Singular_Factors)

% Inputs
% Train_Tensors           : Contain Training patches from all the images
% Singular_Factors        : Low Rank Singular factors from patch modes
%
% Outputs
% 
% Core_Tensor             : Core-tensor obtained from training tensor and Singular Factors
%                           
% Author                  : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update             : 05/04/2018

% %
% Mode-1 = #Images

Number_Modes_Tensor=ndims(Train_Tensors);
    
Aux_Core_Tensor=Train_Tensors;

for i=2:Number_Modes_Tensor
    Aux_Core_Tensor= ttm(Aux_Core_Tensor,Singular_Factors{1,i}',i);
    
end

Core_Tensor=Aux_Core_Tensor;

end

