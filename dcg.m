%% dcg: function description
function score = dcg(true, pred, K)
	score = 0;
	
	for i = 1:K
		score = score + (2^(true(i) == pred(i)) - 1) / (log2(i + 2));
	end

end