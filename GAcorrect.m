function [best_fitness, iterwise_best] = genetic_algorithm(NPop, D, Max_iter, MR, CR, LB, UB)
    population = LB + (UB - LB) * rand(NPop, D);
    fitness = zeros(NPop, 1);  
    iterwise_best = zeros(Max_iter, 1);  
    for i = 1:NPop
        fitness(i) = fit(population(i, :));  
    end
    
    [curr_best, ~] = min(fitness);  

    for iter = 1:Max_iter
        parent1 = population(randi(NPop), :);
        parent2 = population(randi(NPop), :);
        child = zeros(1, D);  
        if rand <= CR
            crossover_point = randi(D);  
            child(1:crossover_point) = parent1(1:crossover_point);
            child(crossover_point+1:end) = parent2(crossover_point+1:end);
        else
            child = parent1;  
        end
        
        if rand <= MR
            mutate_point = randi(D);  
            child(mutate_point) = LB + (UB - LB) * rand;  
        end
        population_pool = [population; child];
        new_fitness = zeros(NPop + 1, 1);
        for i = 1:NPop + 1
            new_fitness(i) = fit(population_pool(i, :));  
        end

        [sortedFitness, Indices] = sort(new_fitness, 'ascend');

        population = population_pool(Indices(1:NPop), :);

        curr_best = sortedFitness(1); 

        iterwise_best(iter) = curr_best;

        fprintf('The best value for iteration %d is: %f\n', iter, curr_best);
    end
    
    best_fitness = curr_best;  
end

function fitness_val = fit(position)
    D = length(position);  
    
    sum_square = sum(position.^2);   
    sum_cos = sum(cos(2 * pi * position));  
    term1 = -20 * exp(-0.2 * sqrt(sum_square / D));
    term2 = -exp(sum_cos / D);
    fitness_val = term1 + term2 + 20 + exp(1);  
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Single run Main

% clear;
% clc;
% 
% NPop = 30;    
% D = 5;        
% Max_iter = 200;  
% MR = 0.1;     
% CR = 0.7;     
% LB = -10;     
% UB = 10;      
% 
% [best_fitness, iterwise_best] = genetic_algorithm(NPop, D, Max_iter, MR, CR, LB, UB);
% 
% disp(['Best fitness found: ', num2str(best_fitness)]);
% 
% figure;
% plot(1:Max_iter, iterwise_best, 'LineWidth', 2);
% xlabel('Iteration');
% ylabel('Best Fitness');
% title('Genetic Algorithm Convergence');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%20 Runs
clear;
clc;

NPop = 100; 
D = 1; 
Max_iter = 200; 
MR = 0.3; 
CR = 0.7; 
LB = -5; 
UB = 5; 
num_runs = 20; 

min_values = zeros(num_runs, Max_iter); 
avg_values = zeros(num_runs, Max_iter); 

for run = 1:num_runs
    [best_fitness, iterwise_best] = genetic_algorithm(NPop, D, Max_iter, MR, CR, LB, UB);
    min_values(run, :) = iterwise_best;
    avg_values(run, :) = mean(min_values(1:run, :), 1); 
end

min_over_runs = min(min_values, [], 1); 

avg_over_runs = mean(min_values, 1); 
std_dev_last_iter = std(min_values(:, Max_iter)); 

figure;
plot(1:Max_iter, min_over_runs, 'b', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Fitness Value');
title('Minimum Fitness Across Runs');
grid on;
legend('Minimum Fitness');

figure;
plot(1:Max_iter, avg_over_runs, 'r', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Fitness Value');
title('Average Fitness Across Runs');
grid on;
legend('Average Fitness');

disp('Best value from all runs:');
disp(min_over_runs(end));

disp('Last iteration average value:');
disp(avg_over_runs(end));

disp('Standard deviation of last iteration values across all runs:');
disp(std_dev_last_iter);
