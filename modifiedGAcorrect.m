function fitness_val = fitnessFunction(x)
    D = length(x);
    fitness_val = sum(x.^2);
end


function [best_fitness, iterwise_best] = modified_genetic_algorithm(MaxIter, D, Npop, CR, MR, LB, UB)
    iterwise_best = zeros(MaxIter, 1);
    best_fitness = inf;
    X = LB + rand(Npop, D) .* (UB - LB);
    fitness = zeros(Npop, 1);

    for i = 1:Npop
        fitness(i) = fitnessFunction(X(i, :));
    end

    for iter = 1:MaxIter
        X_new = X;

        for i = 1:Npop
            r1 = randi(Npop);
            while r1 == i
                r1 = randi(Npop);
            end

            if rand < CR
                crossoverPoint = floor(D / 2);
                X_new(i, 1:crossoverPoint) = X(i, 1:crossoverPoint);
                X_new(i, crossoverPoint+1:end) = X(r1, crossoverPoint+1:end);
            end

            if rand < MR
                mutationPoint = randi(D);
                X_new(i, mutationPoint) = LB(mutationPoint) + rand * (UB(mutationPoint) - LB(mutationPoint));
            end

            fitness_new = fitnessFunction(X_new(i, :));

            if fitness_new < fitness(i)
                X(i, :) = X_new(i, :);
                fitness(i) = fitness_new;
            end
        end

        [sortedFitness, index] = sort(fitness, 'ascend');
        iterwise_best(iter) = sortedFitness(1);
        best_fitness = iterwise_best(iter);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Single Run main
MaxIter = 100;
D = 10;
Npop = 50;
CR = 0.6;
MR = 0.05;
LB = -10 * ones(1, D);
UB = 10 * ones(1, D);

[best_fitness, iterwise_best] = modified_genetic_algorithm(MaxIter, D, Npop, CR, MR, LB, UB);

disp('Final best fitness value from the last iteration:');
disp(best_fitness);

figure;
plot(1:MaxIter, iterwise_best, 'LineWidth', 2, 'Color', 'r');
xlabel('Iteration');
ylabel('Best Fitness Value');
title('Convergence of Best Fitness Over Iterations');
grid on;
legend('Best Fitness');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %20 Runs main

clc;
clear;
MaxIter = 100;
D = 1;
Npop = 50;
CR = 0.6;
MR = 0.05;
n = 20;
LB = -10 * ones(1, D);
UB = 10 * ones(1, D);
min_values = zeros(n, MaxIter); 
avg_values = zeros(n, MaxIter); 

for run = 1:n
    [best_fitness, iterwise_best] = modified_genetic_algorithm(MaxIter, D, Npop, CR, MR, LB, UB);
    min_values(run, :) = iterwise_best;
    avg_values(run, :) = mean(min_values(1:run, :), 1); 
end

min_over_runs = min(min_values, [], 1); 
avg_over_runs = mean(min_values, 1); 
std_dev_last_iter = std(min_values(:, MaxIter)); 

disp('Best value from all runs:');
disp(min_over_runs(end));

disp('Last iteration average value:');
disp(avg_over_runs(end));

disp('Standard deviation of last iteration values across all runs:');
disp(std_dev_last_iter);

figure;
plot(1:MaxIter, min_over_runs, 'b', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Fitness Value');
title('Minimum Fitness Across Runs');
grid on;
legend('Minimum Fitness');

figure;
plot(1:MaxIter, avg_over_runs, 'r', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Fitness Value');
title('Average Fitness Across Runs');
grid on;
legend('Average Fitness');
