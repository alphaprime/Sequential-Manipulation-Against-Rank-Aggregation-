%% idcg: function description
function score = idcg(true, pred, K)
	score = 0;
	
	for i = 1:K
		score = score + 1 / (log2(i + 2));
	end

end