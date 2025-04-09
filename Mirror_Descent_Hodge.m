%% Mirror_Descent: function description
function [Simplex] = Mirror_Descent_Hodge(Theta, Target, C_Matrix, N, No_Iter, r_beta, Lambda)
	M = N*(N-1);

	Theta_tilde = zeros(N, No_Iter);

	simplex = zeros(M, No_Iter+1);

	simplex(:, 1) = ones(M, 1) / M;

	g = zeros(M, No_Iter);

	step_size = zeros(No_Iter, 1);
    
    c =1;

	for t = 1:No_Iter
		
        Theta_tilde(:, t) = RobustEstimation_2(Theta, Target, simplex(:, t), C_Matrix, r_beta, Lambda);
        
		% g(:, t) = Sub_Gradient(Prob_Modelm, Theta, Theta_tilde(:, t));
		
		% g(:, t) = KL_Div(Theta, Theta_tilde(:, t), C_Matrix);

		for i = 1:N
			for j = 1:N
				if i < j
					g((i-1)*(N-1)+j-1, t) = - log((1 + exp(Theta_tilde(j, t) - Theta_tilde(i, t))) / (1 + exp(Theta(j) - Theta(i)))) / (1 + exp(Theta(j) - Theta(i)));
				elseif i > j
					g((i-1)*(N-1)+j, t) = - log((1 + exp(Theta_tilde(j, t) - Theta_tilde(i, t))) / (1 + exp(Theta(j) - Theta(i)))) / (1 + exp(Theta(j) - Theta(i)));
				end
			end
		end
		
		step_size(t) = c / sqrt(t);
		
		tmp = simplex(:, t) .* exp(step_size(t) * g(:, t));
		
		simplex(:, t+1) = tmp ./ sum(tmp);
	end

	Simplex = sum(simplex(:, 2:end), 2) / No_Iter;
end