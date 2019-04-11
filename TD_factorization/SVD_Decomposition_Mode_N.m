% Function to find left singular vectors of a matrix

function [Singular_Factor_Mode_N]=SVD_Decomposition_Mode_N(Target_Tensor,Mode_N,Low_Rank)

% Inputs
% Target_Tensor                : Tensor of Category X
% Mode_N                       : Target Mode of application
% Low_Rank                     : Low_Rank reduction of Mode-N in SVD
% 
% Outputs
% Singular_Factor_Mode_N       : Mode-N Singular factor of Target_Tensor 
% 
% Author                       : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update                  : 08/10/2016

% % 

Matricze_Tensor=(tenmat(Target_Tensor,Mode_N));

[U,~,~]=svds((Matricze_Tensor.data),Low_Rank);

Singular_Factor_Mode_N=U;
    
end
    