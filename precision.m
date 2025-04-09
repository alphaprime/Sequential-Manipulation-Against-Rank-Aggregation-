%% precision: function description
function score = precision(r_true, pred, K)
	true_positive = 0;

	for i = 1:K
		if pred(i) == r_true(i)
			true_positive = true_positive + 1;
		end
	end

	score = true_positive / K;

end