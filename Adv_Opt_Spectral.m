%% Adv_Opt: function description
%% Parameter Description
%% No_item: number of candidates, integer
%% Obs: observed pairwise comparisons, No_item*(No_item-1) vector
%% Target: desired relative ranking score, No_item vector
%% C_Matrix: comparison matrix, No_item*No_item matrix
%% R_alpha: radius of uncertainty set
%% R_beta: radius of support set
%% Eps_Greedy: head probability of flipping a coin

function [Pair_Idx] = Adv_Opt_Spectral(Obs, Target, C_Matrix, No_item, R_alpha, R_beta, Eps_Greedy)

    % iteration of mirror descent
    No_Iter = 100;

    % Line 1 compute the robust estimation based on the previous n ??? 1 comparisons
    % Theta = Optimal_Dual_1(Target, Obs./sum(Obs), C_Matrix, R_alpha, R_beta);
    % Theta = Robust_Estimation(length(Target), Obs./sum(Obs), C_Matrix, R_alpha);
    % Theta = HodgeRank(C_Matrix, ones(length(Obs), 1), Obs');
    % Theta = Spectral_Ranking(Obs, No_item);
    Theta = Worst_Theta(C_Matrix, ones(length(Obs), 1), Obs./sum(Obs), R_alpha, No_Iter);
    
    % Line 2 solve the max-min problem with mirror descent 
    Emp_Simplex = Mirror_Descent_Spectral(Theta, Target, C_Matrix, No_item, No_Iter, R_beta, 0.25);

    % Line 3 balance the exploration and the exploitation by flipping coin with probability Eps_Greedy
    Pair_Idx = Exploration_Exploitation_Balance(Target, Eps_Greedy, Emp_Simplex, No_item);