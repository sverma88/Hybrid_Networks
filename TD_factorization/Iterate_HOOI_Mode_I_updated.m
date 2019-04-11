% HOOI algorithm to redecompose the tensor
% Function to Decompose Tensor to find  Singular factors of Mode I by iteration

function [Singular_Factors]=Iterate_HOOI_Mode_I_updated(Train_Tensor,Singular_Factors,Low_Rank_Mode)

%Input
% Train_Tensor                         : Tensor fof training Patches
% Singular_Factors                     : Singular Mode factors for patch modes
% Low_Rank_Mode                        : Low Rank Reduction required
%
% Output
% Iterated_Factors                     : Iterated Singular Factors after one iteration (HOOI)
%
% Author                               : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update                          : 05/04/2018
%

% MODE-1 = #IMAGES, MODE-2 = #PATCHES

Number_Modes=ndims(Train_Tensor);

for i=2:Number_Modes
   
    
    [Projected_Tensor_Excluded_Modes]=Project_Tensor_Excluding_Mode_I(Train_Tensor,i,Singular_Factors);  
   
    [u]=nvecs(Projected_Tensor_Excluded_Modes,i,Low_Rank_Mode(1,i));
    
    [~,loc] = max(abs(u));
    for j = 1:Low_Rank_Mode(1,i)
        if u(loc(j),j) < 0
            u(:,j) = u(:,j) * -1;
        end
    end
    
    Singular_Factors{1,i}=u;
    
end

end