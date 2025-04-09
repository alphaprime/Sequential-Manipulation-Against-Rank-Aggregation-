%% average_precision: function description
function score = average_precision(r_true, pred, K)
	correct_prediction = 0;

	for i = 1:K
		correct_prediction = correct_prediction + precision(r_true, pred, i) * (r_true(i) == pred(i));
	end
	
	score = correct_prediction / K;
end