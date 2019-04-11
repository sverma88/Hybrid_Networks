%% Function to Compute Decomposition Error of Tensor

function [Total_Error]=Calculate_Decomposition_Error(Train_Tensors,Core_Tensors)

%Input
% Train_Tensors                : Tensor of Training patches
% Core_Tensors                 : Reconstrcuted Tensor
%
%
% Output
% Total_Error                  : Decomposition Error abs(Original Tensor - Reconstructed Tensor) 
%
% Author                       : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update                  : 08/10/2016

%


Total_Error=sqrt(norm(Train_Tensors)^2 - norm(Core_Tensors)^2);

end