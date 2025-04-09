%% Mirror_Descent: function description
function [Simplex] = Mirror_Descent_Spectral(Theta, Target, C_Matrix, N, No_Iter, r_beta, Lambda)
	M = N*(N-1);

	max_Theta = max(Theta);
	min_Theta = min(Theta);

	Theta_scale = (Theta - min_Theta) / (max_Theta - min_Theta);

	Theta_tilde = zeros(N, No_Iter);

	simplex = zeros(M, No_Iter+1);

	simplex(:, 1) = ones(M, 1) / M;

	g = zeros(M, No_Iter);

	step_size = zeros(No_Iter, 1);
    
    c = 1;

	for t = 1:No_Iter
		
        Theta_tilde(:, t) = RobustEstimation_2(Theta, Target, simplex(:, t), C_Matrix, r_beta, Lambda);

        max_tilde = max(Theta_tilde(:, t));
        min_tilde = min(Theta_tilde(:, t));

        tmp_theta = (Theta_tilde(:, t) - min_tilde) / (max_tilde - min_tilde);

        % tmp_theta = Theta_tilde(:, t);
        
		% g(:, t) = Sub_Gradient(Prob_Modelm, Theta, Theta_tilde(:, t));
		
		% g(:, t) = KL_Div(Theta, Theta_tilde(:, t), C_Matrix);

		for i = 1:N
			for j = 1:N
				if i < j
					g((i-1)*(N-1)+j-1, t) = - log10((1 + exp(tmp_theta(j) - tmp_theta(i))) / (1 + exp(Theta_scale(j) - Theta_scale(i)))) / (1 + exp(Theta_scale(j) - Theta_scale(i)));
				elseif i > j
					g((i-1)*(N-1)+j, t) = - log10((1 + exp(tmp_theta(j) - tmp_theta(i))) / (1 + exp(Theta_scale(j) - Theta_scale(i)))) / (1 + exp(Theta_scale(j) - Theta_scale(i)));
				end
			end
		end
		
		step_size(t) = c / sqrt(t);
		
		tmp = simplex(:, t) .* exp(step_size(t) * g(:, t));
		
		simplex(:, t+1) = tmp ./ sum(tmp);
	end

	Simplex = sum(simplex(:, 2:end), 2) / No_Iter;
end