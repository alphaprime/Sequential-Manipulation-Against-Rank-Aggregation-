%% ndcg: function description
function score = ndcg(r_true, pred, K)
	
	dcg_score = dcg(r_true, pred, K);
	idcg_score = idcg(r_true, pred, K);
	score = dcg_score / idcg_score;

end