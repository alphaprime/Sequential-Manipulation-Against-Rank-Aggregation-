close all;
clear;
clc;

% Number of Candidates
n = 10;
N = n * (n-1);
% Number of Top Winner
top_K = 3;
% Inconsistency Ratio
ratio_outlier = 0.1;

% Uncertainty Radius 1 for weight
% R_alpha = [0.1, 0.01, 0.001];
R_alpha = 0.001;

% Uncertainty Radius 2 for score
% R_beta = [0.1, 0.01, 0.001];
R_beta = 0.001;

% Number of Trails
T = 50;

% Relative cost
% Chi = [0.1, 0.01, 0.001];
Chi = 0.001;

% Balance between Exploration and Exploitation
Eps_Greedy = 0.2 * abs(log10(Chi)).^(-0.25);
% Eps_Greedy = 0;

% Attack occurs when sampled_ratio% data has been sampled
% sampled_ratio = [0.85, 0.9, 0.95];
sampled_ratio = 0.9;

% Knowledge Level 
% knowledge_ratio = [0.7, 0.8, 0.9];
knowledge_ratio = 0.9;

% stop_thred = [5, 10, 20];
stop_thred = 2;

% Ranking result without perturbation
[A, y, W, true_r]        = data_generation(n, ratio_outlier);
Offline_Spectral_Theta      = Spectral_Ranking(W, n);
Offline_Spectral_Ranking    = score_to_ranking(Offline_Spectral_Theta);
Offline_Spectral_Score      = eval_ranking(true_r', Offline_Spectral_Ranking, top_K);

Greedy_List = zeros(top_K, n-1);
Simple_List = zeros(top_K, 0.5 * N);

Seq_Adv = cell(T, 1);
Seq_Bak = cell(T, 1);
Seq_Gdy = cell(T, 1);
Seq_Smp = cell(T, 1);
Seq_Rnd = cell(T, 1);

Rnd_Spectral_Theta = cell(T, 1);
Smp_Spectral_Theta = cell(T, 1);
Gdy_Spectral_Theta = cell(T, 1);
Adv_Spectral_Theta = cell(T, 1);

Rnd_Spectral_Ranking = cell(T, 1); 
Smp_Spectral_Ranking = cell(T, 1);
Gdy_Spectral_Ranking = cell(T, 1);
Adv_Spectral_Ranking = cell(T, 1);

Rnd_Spectral_Score = cell(T, 1);
Smp_Spectral_Score = cell(T, 1);
Gdy_Spectral_Score = cell(T, 1);
Adv_Spectral_Score = cell(T, 1);

Rnd_Spectral_Theta_Per_Turn = cell(T, 1);
Smp_Spectral_Theta_Per_Turn = cell(T, 1);
Gdy_Spectral_Theta_Per_Turn = cell(T, 1);
Adv_Spectral_Theta_Per_Turn = cell(T, 1);

Rnd_Spectral_Ranking_Per_Turn = cell(T, 1); 
Smp_Spectral_Ranking_Per_Turn = cell(T, 1);
Gdy_Spectral_Ranking_Per_Turn = cell(T, 1);
Adv_Spectral_Ranking_Per_Turn = cell(T, 1);

Rnd_Spectral_Score_Per_Turn = cell(T, 1);
Smp_Spectral_Score_Per_Turn = cell(T, 1);
Gdy_Spectral_Score_Per_Turn = cell(T, 1);
Adv_Spectral_Score_Per_Turn = cell(T, 1);

% Generate the goal 
Tmp_Ranking = [8;9;10;7;5;6;4;3;2;1];
for j = 1:length(Tmp_Ranking)
    % Re-assign the latent preference score
    Score_Target(Tmp_Ranking(j)) = Offline_Spectral_Theta(Offline_Spectral_Ranking(j));
end
if size(Score_Target, 1) == 1
    Score_Target = Score_Target';
end
max_Score = max(Score_Target);
min_Score = min(Score_Target);
Score_Target = (Score_Target - min_Score) / (max_Score - min_Score);
Rank_Target = Tmp_Ranking;
cnt = 1;
for l_id = 1:n-1
    for r_id = l_id+1:n
        if l_id == 1
            if Rank_Target(l_id) < Rank_Target(r_id)
                Greedy_List(cnt) = (Rank_Target(1) - 1) * (n-1) + Rank_Target(r_id) - 1;
            else
                Greedy_List(cnt) = (Rank_Target(1) - 1) * (n-1) + Rank_Target(r_id);
            end
        end
        if Rank_Target(l_id) < Rank_Target(r_id)
            Simple_List(cnt) = (Rank_Target(l_id) - 1) * (n-1) + Rank_Target(r_id) - 1;
        else
            Simple_List(cnt) = (Rank_Target(l_id) - 1) * (n-1) + Rank_Target(r_id);
        end
        cnt = cnt + 1;
    end
end

for t = 1:T
    Sum_W = sum(W);
    Seq_W = randperm(Sum_W, Sum_W);
    Cum_W = cumsum(W);
    for j = 1:Sum_W
        id = min(find(Cum_W >= Seq_W(j)));
        Seq_id(j) = id;
    end
    Sampled_data = Seq_id(1:floor(sampled_ratio * Sum_W));
    Par_W = length(Sampled_data);
    Turn = Sum_W - Par_W;
    Adv_mask = rand(1, Par_W);
    Adv_mask(Adv_mask > knowledge_ratio) = 0;
    Adv_mask(find(Adv_mask)) = 1;
    Tmp_Seq_Adv = Adv_mask .* Sampled_data;
    Tmp_Seq_Rnd = Tmp_Seq_Adv;
    Tmp_Seq_Smp = Tmp_Seq_Adv;
    Tmp_Seq_Gdy = Tmp_Seq_Adv;
    Tmp_Seq_Bak = zeros(1, Turn);
    [tmp_count, tmp_idx] = hist(Tmp_Seq_Adv, unique(Tmp_Seq_Adv));
    Adv_W(tmp_idx(2:end)) = tmp_count(2:end);
    stop_time = 1;
    action_flag = 1;
    for s = 1:Turn
        % Adversarial Action
        while stop_time < stop_thred && action_flag == 1
            Adv_ID = Adv_Opt_Spectral(Adv_W, Score_Target, A, n, R_alpha, R_beta, Eps_Greedy);
            % Adv_W(Adv_ID) = Adv_W(Adv_ID) + 1;
            Tmp_Seq_Adv(end+1) = Adv_ID;
            Tmp_Seq_Rnd(end+1) = randi(N, 1);
            Tmp_Seq_Smp(end+1) = Simple_List(randi(0.5*N, 1));
            Tmp_Seq_Gdy(end+1) = Greedy_List(randi(n-1, 1));
            % update stop time
            stop_time = stop_time + 1;
        end

        Tmp_Seq_Adv_per_Turn = [Tmp_Seq_Adv, Seq_id(Par_W+s)];
        Tmp_Seq_Rnd_per_Turn = [Tmp_Seq_Rnd, Seq_id(Par_W+s)];
        Tmp_Seq_Smp_per_Turn = [Tmp_Seq_Smp, Seq_id(Par_W+s)];
        Tmp_Seq_Gdy_per_Turn = [Tmp_Seq_Gdy, Seq_id(Par_W+s)];

        [tmp_count, tmp_idx]  = hist(Tmp_Seq_Rnd_per_Turn, unique(Tmp_Seq_Rnd_per_Turn));
        Rnd_W_Turn(tmp_idx(2:end)) = tmp_count(2:end);
        
        [tmp_count, tmp_idx]  = hist(Tmp_Seq_Smp_per_Turn, unique(Tmp_Seq_Smp_per_Turn));
        Smp_W_Turn(tmp_idx(2:end)) = tmp_count(2:end);
        
        [tmp_count, tmp_idx]  = hist(Tmp_Seq_Gdy_per_Turn, unique(Tmp_Seq_Gdy_per_Turn));
        Gdy_W_Turn(tmp_idx(2:end)) = tmp_count(2:end);
        
        [tmp_count, tmp_idx]  = hist(Tmp_Seq_Adv_per_Turn, unique(Tmp_Seq_Adv_per_Turn));
        Adv_W_Turn(tmp_idx(2:end)) = tmp_count(2:end);
        
        Rnd_Theta_Turn(s, :)   = Spectral_Ranking(Rnd_W_Turn, n);
        Rnd_Ranking_Turn(s, :) = score_to_ranking(Rnd_Theta_Turn(s, :));
        Rnd_Score_Turn(s, :)   = eval_ranking(Rank_Target, Rnd_Ranking_Turn(s, :)', top_K);
        
        Smp_Theta_Turn(s, :)   = Spectral_Ranking(Smp_W_Turn, n);
        Smp_Ranking_Turn(s, :) = score_to_ranking(Smp_Theta_Turn(s, :));
        Smp_Score_Turn(s, :)   = eval_ranking(Rank_Target, Smp_Ranking_Turn(s, :)', top_K);
        
        Gdy_Theta_Turn(s, :)   = Spectral_Ranking(Gdy_W_Turn, n);
        Gdy_Ranking_Turn(s, :) = score_to_ranking(Gdy_Theta_Turn(s, :));
        Gdy_Score_Turn(s, :)   = eval_ranking(Rank_Target, Gdy_Ranking_Turn(s, :)', top_K);
        
        Adv_Theta_Turn(s, :)   = Spectral_Ranking(Adv_W_Turn, n);
        Adv_Ranking_Turn(s, :) = score_to_ranking(Adv_Theta_Turn(s, :));
        Adv_Score_Turn(s, :)   = eval_ranking(Rank_Target, Adv_Ranking_Turn(s, :)', top_K);

        [tmp_count, tmp_idx]  = hist(Tmp_Seq_Rnd, unique(Tmp_Seq_Rnd));
        Rnd_W_bak(tmp_idx(2:end)) = tmp_count(2:end);
        
        [tmp_count, tmp_idx]  = hist(Tmp_Seq_Smp, unique(Tmp_Seq_Smp));
        Smp_W_bak(tmp_idx(2:end)) = tmp_count(2:end);
        
        [tmp_count, tmp_idx]  = hist(Tmp_Seq_Gdy, unique(Tmp_Seq_Gdy));
        Gdy_W_bak(tmp_idx(2:end)) = tmp_count(2:end);
        
        [tmp_count, tmp_idx]  = hist(Tmp_Seq_Adv, unique(Tmp_Seq_Adv));
        Adv_W_bak(tmp_idx(2:end)) = tmp_count(2:end);

        W_Adv_per_Turn(s, :) = Adv_W_bak;
        W_Rnd_per_Turn(s, :) = Rnd_W_bak;
        W_Smp_per_Turn(s, :) = Smp_W_bak;
        W_Gdy_per_Turn(s, :) = Gdy_W_bak;

        p = rand;
        if p < knowledge_ratio
            % update knowledge
            Tmp_Seq_Adv(end+1) = Seq_id(Par_W+s);
            Tmp_Seq_Rnd(end+1) = Seq_id(Par_W+s);
            Tmp_Seq_Smp(end+1) = Seq_id(Par_W+s);
            Tmp_Seq_Gdy(end+1) = Seq_id(Par_W+s);
            Adv_W(Seq_id(Par_W+s)) = Adv_W(Seq_id(Par_W+s)) + 1;
            Tmp_Seq_Bak(s) = 0;
            action_flag = 1;
        else
            Tmp_Seq_Bak(s) = Seq_id(Par_W+s);
            action_flag = 0;
        end
        stop_time = 1;
    end

    Rnd_Spectral_Theta_Per_Turn{t} = Rnd_Theta_Turn;
    Smp_Spectral_Theta_Per_Turn{t} = Smp_Theta_Turn;
    Gdy_Spectral_Theta_Per_Turn{t} = Gdy_Theta_Turn;
    Adv_Spectral_Theta_Per_Turn{t} = Adv_Theta_Turn;  

    Rnd_Spectral_Ranking_Per_Turn{t} = Rnd_Ranking_Turn;  
    Smp_Spectral_Ranking_Per_Turn{t} = Smp_Ranking_Turn; 
    Gdy_Spectral_Ranking_Per_Turn{t} = Gdy_Ranking_Turn; 
    Adv_Spectral_Ranking_Per_Turn{t} = Adv_Ranking_Turn;     

    Rnd_Spectral_Score_Per_Turn{t} = Rnd_Score_Turn;
    Smp_Spectral_Score_Per_Turn{t} = Smp_Score_Turn;
    Gdy_Spectral_Score_Per_Turn{t} = Gdy_Score_Turn;
    Adv_Spectral_Score_Per_Turn{t} = Adv_Score_Turn;

    Rnd_Spectral_Seq_Per_Turn{t} = W_Rnd_per_Turn;
    Smp_Spectral_Seq_Per_Turn{t} = W_Smp_per_Turn;
    Gdy_Spectral_Seq_Per_Turn{t} = W_Gdy_per_Turn;
    Adv_Spectral_Seq_Per_Turn{t} = W_Adv_per_Turn;

    Seq_Rnd{t} = Tmp_Seq_Rnd;
    Tmp_Seq_Rnd = [Tmp_Seq_Rnd, Tmp_Seq_Bak];
    [tmp_count, tmp_idx]  = hist(Tmp_Seq_Rnd, unique(Tmp_Seq_Rnd));
    Rnd_W(tmp_idx(2:end)) = tmp_count(2:end);

    Seq_Smp{t} = Tmp_Seq_Smp;
    Tmp_Seq_Smp = [Tmp_Seq_Smp, Tmp_Seq_Bak];
    [tmp_count, tmp_idx]  = hist(Tmp_Seq_Smp, unique(Tmp_Seq_Smp));
    Smp_W(tmp_idx(2:end)) = tmp_count(2:end);

    Seq_Gdy{t} = Tmp_Seq_Gdy;
    Tmp_Seq_Gdy = [Tmp_Seq_Gdy, Tmp_Seq_Bak];
    [tmp_count, tmp_idx]  = hist(Tmp_Seq_Gdy, unique(Tmp_Seq_Gdy));
    Gdy_W(tmp_idx(2:end)) = tmp_count(2:end);

    Seq_Adv{t} = Tmp_Seq_Adv;
    Tmp_Seq_Adv = [Tmp_Seq_Adv, Tmp_Seq_Bak];
    [tmp_count, tmp_idx]  = hist(Tmp_Seq_Adv, unique(Tmp_Seq_Adv));
    Adv_W(tmp_idx(2:end)) = tmp_count(2:end);

    Seq_Bak{t} = Tmp_Seq_Bak;

    % Evaluation
    Rnd_Theta   = Spectral_Ranking(Rnd_W, n);
    Rnd_Ranking = score_to_ranking(Rnd_Theta);
    Rnd_Score   = eval_ranking(Rank_Target, Rnd_Ranking, top_K);
    Rnd_Spectral_Theta{t}   =  Rnd_Theta;   
    Rnd_Spectral_Ranking{t} =  Rnd_Ranking; 
    Rnd_Spectral_Score{t}   =  Rnd_Score; 

    Smp_Theta   = Spectral_Ranking(Smp_W, n);
    Smp_Ranking = score_to_ranking(Smp_Theta);
    Smp_Score   = eval_ranking(Rank_Target, Smp_Ranking, top_K);
    Smp_Spectral_Theta{t}   =  Smp_Theta;
    Smp_Spectral_Ranking{t} =  Smp_Ranking;
    Smp_Spectral_Score{t}   =  Smp_Score;

    Gdy_Theta   = Spectral_Ranking(Gdy_W, n);
    Gdy_Ranking = score_to_ranking(Gdy_Theta);
    Gdy_Score   = eval_ranking(Rank_Target, Gdy_Ranking, top_K);
    Gdy_Spectral_Theta{t}   =  Gdy_Theta;
    Gdy_Spectral_Ranking{t} =  Gdy_Ranking;
    Gdy_Spectral_Score{t}   =  Gdy_Score;

    Adv_Theta   = Spectral_Ranking(Adv_W, n);
    Adv_Ranking = score_to_ranking(Adv_Theta);
    Adv_Score   = eval_ranking(Rank_Target, Adv_Ranking, top_K);
    Adv_Spectral_Theta{t}   =  Adv_Theta;
    Adv_Spectral_Ranking{t} =  Adv_Ranking;
    Adv_Spectral_Score{t}   =  Adv_Score;
    disp('Finish Attack.')
end

save('simulation_spectral_230212_v4.mat');