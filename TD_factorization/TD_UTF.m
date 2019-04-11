%% Function to decompose multiple labeled tensors using CUTF algorithm

function [Iterated_Factors]=TD_UTF(Train_Tensor,Low_Rank_Modes,Error_Threshold,Max_iterations)

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

    
Initial_Factors=Tensor_Decomposition_HOSVD(Train_Tensor,Low_Rank_Modes);
    

Iteration_Count=1;         Set Iterations Count

Iterated_Factors=Initial_Factors;    
% HOOI Iteration for Converging Singular Factors obtained Above
fittold = 0;
fitchange = -5;

while( (fitchange > Error_Threshold ) && ( Iteration_Count <= Max_iterations ))       
    
    %HOOI on Patch Modes only
    Iterated_Singular_Factors=Iterate_HOOI_Mode_I_updated(Train_Tensor,Iterated_Factors,Low_Rank_Modes);
   
      
    % Reconstructing Tensor
    [Core_Tensors]=Core_Tensor(Train_Tensor,Iterated_Singular_Factors);
    
    % Calculating New Error
    
    Error_This_iteration=Calculate_Decomposition_Error(Train_Tensor,Core_Tensors);
    
    % Throwing on termial to keep a track
    normresidual=abs(Previous_Iteration_Error-Error_This_iteration);
    Iteration_Count=Iteration_Count+1;
    fit = 1 - (normresidual / norm(Train_Tensor)); %fraction explained by model
    fitchange = abs(fittold - fit);
    
    fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n', Iteration_Count, fit, fitchange);

    
    % Updating Error for next iteration
    fittold=fit;
    Iterated_Factors = Iterated_Singular_Factors;
    
end



end

