%% Construct_Transition_Matrix: function description
function [A] = Construct_Transition_Matrix(w, n)
	A = zeros(n);
	right_id = 1;
	left_id = 2;
	for idx = 1:length(w)
		A(left_id, right_id) = w(idx);
		if left_id < n
			left_id = left_id + 1;
			if left_id == right_id
				left_id = left_id + 1;
			end
		else
			left_id = 1;
			right_id = right_id + 1;
		end
    end
end