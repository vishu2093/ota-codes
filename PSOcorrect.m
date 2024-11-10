% NPop = 30;        
% D = 5;            
% Max_iter = 200;   
% LB = -10 * ones(1, D);  
% UB = 10 * ones(1, D);   
% 
% iterwise_best = particleSwarmOptimization(NPop, Max_iter, D, LB, UB);
% 
% fprintf("Best value of fitness for the function is: %f\n", iterwise_best(end));
% 
% figure;
% plot(1:Max_iter, iterwise_best, 'LineWidth', 2);
% xlabel('Iteration');
% ylabel('Best Fitness');
% title('PSO Convergence');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%20 runs main
clc;
clear;

MaxIter = 100; 
D = 5; 
Npop = 50; 
CR = 0.6; 
MR = 0.05; 
n = 20; 
LB = -10 * ones(1, D); 
UB = 10 * ones(1, D); 

PSO_bestFitness = zeros(MaxIter, n);
PSO_meanFitness = zeros(MaxIter, 1);

for run = 1:n
    fprintf("Beginning run %d.", run);
    [bestFitnessOverTime] = particleSwarmOptimization(Npop, MaxIter, D, LB, UB);
    PSO_bestFitness(:, run) = bestFitnessOverTime;
end

PSO_meanFitness = mean(PSO_bestFitness, 2);
PSO_stddevFitness = std(PSO_bestFitness, 0, 2);

figure;
plot(1:MaxIter, PSO_meanFitness, 'b', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Mean Fitness Value');
title('Mean Fitness Value Over Iterations');
legend('PSO');
grid on;

figure;
plot(1:MaxIter, min(PSO_bestFitness, [], 2), 'b', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness Value');
title('Best Fitness Value Over Iterations');
legend('PSO');
grid on;

disp('Best value from all runs:');
disp(PSO_bestFitness(end));

disp('Last iteration average value:');
disp(PSO_meanFitness(end));

disp('Standard deviation of last iteration values across all runs:');
disp(PSO_stddevFitness(end));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bestFitnessOverTime] = particleSwarmOptimization(nParticles, maxIterations, nDimensions, LB, UB)
    cognitiveCoeff = 2; 
    socialCoeff = 2;    
    inertiaWeight = 0.5; 

    positions = LB + (UB - LB) .* rand(nParticles, nDimensions);
    velocities = zeros(nParticles, nDimensions);

    fitness = zeros(nParticles, 1);
    for i = 1:nParticles
        fitness(i) = fitnessFunction(positions(i,:));
    end

    pBestPositions = positions;
    pBestFitness = fitness;

    [globalBestFitness, globalBestIndex] = min(fitness);
    gBestPosition = positions(globalBestIndex, :);

    bestFitnessOverTime = zeros(maxIterations, 1);

    for iteration = 1:maxIterations
        r1 = rand(nParticles, nDimensions); 
        r2 = rand(nParticles, nDimensions);

        velocities = inertiaWeight * velocities + ...
            cognitiveCoeff * r1 .* (pBestPositions - positions) + ...
            socialCoeff * r2 .* (gBestPosition - positions);

        positions = positions + velocities;
        positions = max(min(positions, UB), LB);

        fitness = zeros(nParticles, 1);
        for i = 1:nParticles
            fitness(i) = fitnessFunction(positions(i,:));
        end

        improved = fitness < pBestFitness;
        pBestPositions(improved, :) = positions(improved, :);
        pBestFitness(improved) = fitness(improved);

        [currentGlobalBestFitness, globalBestIndex] = min(fitness);
        if currentGlobalBestFitness < globalBestFitness
            globalBestFitness = currentGlobalBestFitness;
            gBestPosition = positions(globalBestIndex, :);
        end
        fprintf('Fitness value for iteration %d is: %f\n', iteration, globalBestFitness);
        bestFitnessOverTime(iteration) = globalBestFitness;
    end
end

function fitness_val = fitnessFunction(x)
    D = numel(x);  
    sum_square = sum(x.^2);  
    sum_cos = sum(cos(2 * pi * x));  
    fitness_val = -20 * exp(-0.2 * sqrt(sum_square / D)) - exp(sum_cos / D) + 20 + exp(1);  
end
