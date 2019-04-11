% Function to decompose given tensor using HOSVD algorithm 


function [Singular_Factors]=Tensor_Decomposition_HOSVD(Train_Tensor,Low_Rank)

%Input
% Train_Tensor            : Tensor of training Imagess 
% Low_Rank                : Low Rank reduction requied in SVD
% Indices                 : row vector containing indices of modes to decompose (OPTIONAL ARGUMENT)
% 
%Output
% Singular_Factors        : Cell of size Indices, contains Singular Factors of each mode 
% 
% Author                  : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update             : 05/04/2018

% % 

% Decomposing Tensor into lower rank using HOSVD Algorithm
% The first Mode is Number of Samples

Number_Modes=ndims(Train_Tensor);
Singular_Factors=cell(1,Number_Modes);


%Decomposing modes using HOSVD
for i=2:Number_Modes
        u = nvecs(Train_Tensor,i,Low_Rank(1,i));        
%         [val,loc] = max(abs(u));
%         for j = 1:Low_Rank(1,i)
%             if u(loc(j),j) < 0
%                 u(:,j) = u(:,j) * -1;
%             end
%         end
        
        Singular_Factors{1,i} = u;
        
        
end

end