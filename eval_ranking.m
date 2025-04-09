%% eval_ranking: function description
function score = eval_ranking(r_true, pred, K)
	score = zeros(5, 1);
	% Kendall tau 
	score(1) = corr(r_true, pred, 'type', 'Kendall');
	% Reciprocal Rank
	score(2) = reciprocal_rank(r_true, pred);
	% Precision
	score(3) = precision(r_true, pred, K);
	% Average Precision
	score(4) = average_precision(r_true, pred, K);
	% NDCG
	score(5) = ndcg(r_true, pred, K);
end