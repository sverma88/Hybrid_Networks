% Function to Project Tensor on all Modes excluding specificy Modes "I"

function [Projected_Tensor_Exclude_Mode_I]=Project_Tensor_Excluding_Mode_I(Target_Tensor,Exclude_Modes,Tensor_Singular_Factors)

%Input
% Target_Tensor                   : Tensor of Category X
% Exclude_Modes                   : row vector of specific modes on which Target_Tensor 
%                                   will not be Projetced
% Tensor_Singular_Factors         : Singular Factors of Target_Tensor
%
%
% Output
% Projected_Tensor_Exclude_Mode_I : Tensor 'A' multiplied by Specific modes Factors:
%                                   multilinear tensor product excluding modes "Exclude_Modes"
%
% Author                          : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update                     : 24/04/2017

%


Projected_Tensor_Exclude_Mode_I=Target_Tensor;
Number_Modes=ndims(Target_Tensor);
Exclude_Modes = [1 Exclude_Modes];

for j=1:Number_Modes
    
    if(~any(j==Exclude_Modes))
       
        Projected_Tensor_Exclude_Mode_I=ttm(Projected_Tensor_Exclude_Mode_I,Tensor_Singular_Factors{1,j}',j);
        
    end
    
end

end