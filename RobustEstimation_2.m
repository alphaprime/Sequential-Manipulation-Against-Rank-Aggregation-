function [theta, loss, penalty] = RobustEstimation_2(est, target, cate_dist, C, Beta, Lambda)
	n = length(target);
    N = n*(n-1);
    T = 50;
	Theta = ones(n, T);
	Loss = zeros(T, 1);
	Penalty = zeros(T, 1);
	grad_old = zeros(n, 1);
	step_size = 0.0001 * ones(T, 1);
	est_ranking = score_to_ranking(est);
	theta_ranking = est_ranking;

	% while est_ranking == theta_ranking
		for t = 1:T
			grad_loss = zeros(n, 1);
			grad_pelty = zeros(n, 1);
			for i = 1:n
				for j = 1:n
					if i < j
						grad_loss(i) = grad_loss(i) - cate_dist((i-1)*(n-1)+j-1) * (exp(Theta(j, t)- Theta(i, t))) / (1 + exp(Theta(j, t)- Theta(i, t)))^3;
					elseif i > j
						grad_loss(i) = grad_loss(i) - cate_dist((i-1)*(n-1)+j)   * (exp(Theta(j, t)- Theta(i, t))) / (1 + exp(Theta(j, t)- Theta(i, t)))^3;
					end
				end
			end
			grad_pelty = 4 * (norm(Theta(:, t)-target)^2-Beta) * (Theta(:, t)-target);
			grad_new = grad_loss + Lambda * grad_pelty;	

			Loss(t) = sum(cate_dist ./ (1 + exp(- C * est)) .* log ((1 + exp(- C * Theta(:, t))) ./ (1 + exp(- C * est))));
			
			Penalty(t) = Lambda * (norm(Theta(:, t)-target)^2-Beta)^2;	

			if norm(grad_new) > 1e-6
				if t > 1
					delta_theta = Theta(:, t) - Theta(:, t-1);
					delta_grad = grad_new - grad_old;
					step_size(t) = (norm(delta_theta)^2 + 1e-8) / (delta_theta' * delta_grad + 1e-8);
				end		

				if t < T
					Theta(:, t+1) = Theta(:, t) - step_size(t) * grad_new;
				end	

				grad_old = grad_new;
			else
				break;
			end
		end
	% 	theta_ranking = score_to_ranking(Theta(:, t));
	% end

	theta = Theta(:, t);
	loss = Loss(t);
	penalty = Penalty(t);
end