%% Function to decompose multiple labeled tensors using CUTF algorithm

function [Iterated_Factors,TXmean]=TD_UTF(Train_Tensor,Low_Rank_Modes,Error_Threshold,Max_iterations)

%Input
% Train_Tensor                   : Tensor of all patches
% Low_Rank_Modes                 : Low Rank Reductions required for Modes
% Error_Threshold                : Allowable Error tolerance limit for
%                                  Termination of Decompositon algorithm
% Max_iterations                 : Maximum allowable iterations limit for
%                                  Termination of Decompositon algorithm
% Indices                        : Row-Vector of Indices to Factorize
%
% Output
% Iterated_Factors               : Cell containing Converged  Singular Factors
%
%
% Author                         : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update                    : 05/04/2018


%%

% Remembere  mode-1 is #Images and mode-2 is #patches
% Find Singular Factors of patches

TXmean = mean(double(Train_Tensor),1);
N = ndims(Train_Tensor)-1;
numSpl = size(Train_Tensor,1);
Train_Tensor = tensor(double(Train_Tensor) - repmat(TXmean,[numSpl, ones(1,N)]));
Initial_Factors=Tensor_Decomposition_HOSVD(Train_Tensor,Low_Rank_Modes);
    
% Finding Core Tensors and reconstructed Tensors
[Core_Tensors]=Core_Tensor(Train_Tensor,Initial_Factors);
% [Reconstructed_Tensors]=Reconstructed_Tensor(Core_Tensors,Initial_Factors);

% Decomposition_Error=Calculate_Decomposition_Error(Train_Tensor,Reconstructed_Tensors);
Decomposition_Error=Calculate_Decomposition_Error(Train_Tensor,Core_Tensors);
fitchange=Decomposition_Error;                % Set Decomposition Error Checking
Iteration_Count=1;        
fittold = 0;

fprintf(' Iter %2d: error = %e \n', 1, Decomposition_Error);


Iterated_Factors=Initial_Factors;                      
% HOOI Iteration for Converging Singular Factors obtained Above

while( (fitchange > Error_Threshold ) && ( Iteration_Count <= Max_iterations ))       
    
    %HOOI on Patch Modes only
    Iterated_Singular_Factors=Iterate_HOOI_Mode_I_updated(Train_Tensor,Iterated_Factors,Low_Rank_Modes);
   
      
    % Reconstructing Tensor
    [Core_Tensors]=Core_Tensor(Train_Tensor,Iterated_Singular_Factors);
   
    % Calculating New Error    
    normresidual=Calculate_Decomposition_Error(Train_Tensor,Core_Tensors);
    fit = 1-(normresidual / norm(Train_Tensor));
    fitchange = abs(fittold - fit);
    
    
    % Throwing on termial to keep a track
    Iteration_Count=Iteration_Count+1;
    fittold = fit;
    
    fprintf(' Iter %2d: error = %e change = %7.1e\n', Iteration_Count, fit, fitchange);
    
    % Updating Error for next iteration
    Iterated_Factors = Iterated_Singular_Factors;
    
    
    
end



end

