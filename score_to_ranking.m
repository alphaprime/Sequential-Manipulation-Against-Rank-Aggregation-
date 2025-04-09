%% score_to_ranking: function description
function idx = score_to_ranking(theta)
    n = length(theta);
	[~, idx] = sort(theta, 'descend');
	% r = 1:n;
	% r(idx) = r;
end