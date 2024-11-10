clc;
clear;

NPop = 30;            
D = 5;                
MaxIter = 200;        
LB = -10 * ones(1, D); 
UB = 10 * ones(1, D);

fprintf("Running PSO...\n");
[PSO_bestFitness] = particleSwarmOptimization(NPop, D, MaxIter, @fitnessFunction, LB, UB);
fprintf("Best fitness for PSO: %f\n", PSO_bestFitness(end));

fprintf("Running APSO...\n");
[APSO_bestFitness] = acceleratedPSO(NPop, MaxIter, D, @fitnessFunction, LB, UB, 0.5, 1);
fprintf("Best fitness for APSO: %f\n", APSO_bestFitness(end));

figure;

subplot(2, 1, 1);
plot(1:MaxIter, PSO_bestFitness, 'b', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness');
title('PSO Convergence');
grid on;
subplot(2, 1, 2);
plot(1:MaxIter, APSO_bestFitness, 'r', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness');
title('APSO Convergence');
grid on;

fprintf("Final Best Fitness from PSO: %f\n", PSO_bestFitness(end));
fprintf("Final Best Fitness from APSO: %f\n", APSO_bestFitness(end));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 20 runs main
% clc;
% clear;
% 
% MaxIter = 100; 
% n = 20; 
% D = 3;  
% LB = -10 * ones(1, D);
% UB = 10 * ones(1, D);
% 
% PSO_bestFitness = zeros(MaxIter, n);
% accPSO_bestFitness = zeros(MaxIter, n);
% 
% for run = 1:n
%     fprintf("Beginning run %d\n", run);
% 
%     [bestFitnessOverTime] = particleSwarmOptimization(D, MaxIter, @fitnessFunction, LB, UB);
%     PSO_bestFitness(:, run) = bestFitnessOverTime;
%     fprintf("Run %d - PSO: Best Fitness at last iteration: %f\n", run, bestFitnessOverTime(end));
% 
%     [bestFitnessOverTime] = acceleratedPSO(D, MaxIter, @fitnessFunction, LB, UB, 0.5, 1);
%     accPSO_bestFitness(:, run) = bestFitnessOverTime;
%     fprintf("Run %d - APSO: Best Fitness at last iteration: %f\n", run, bestFitnessOverTime(end));
% end
% 
% PSO_meanFitness = mean(PSO_bestFitness, 2);
% PSO_stddevFitness = std(PSO_bestFitness, 0, 2);
% accPSO_meanFitness = mean(accPSO_bestFitness, 2);
% accPSO_stddevFitness = std(accPSO_bestFitness, 0, 2);
% 
% figure;
% plot(1:MaxIter, min(PSO_bestFitness, [], 2), 'b', 'LineWidth', 2);  
% hold on;
% plot(1:MaxIter, min(accPSO_bestFitness, [], 2), 'r--', 'LineWidth', 2);  
% xlabel('Iteration');
% ylabel('Best Fitness Value');
% title('Best Fitness Value Over Iterations');
% legend('PSO', 'APSO');
% grid on;
% 
% figure;
% plot(1:MaxIter, PSO_meanFitness, 'b', 'LineWidth', 2);  
% hold on;
% plot(1:MaxIter, accPSO_meanFitness, 'r--', 'LineWidth', 2); 
% xlabel('Iteration');
% ylabel('Mean Fitness Value');
% title('Mean Fitness Value Over Iterations');
% legend('PSO', 'APSO');
% grid on;
% 
% fprintf('\n--- Final Results ---\n');
% fprintf('PSO Best Value from all runs at last iteration: %f\n', min(PSO_bestFitness(MaxIter, :), [], 2));
% fprintf('PSO Last Iteration Mean Value: %f\n', PSO_meanFitness(end));
% fprintf('PSO Standard Deviation at Last Iteration: %f\n', PSO_stddevFitness(end));
% 
% fprintf('APSO Best Value from all runs at last iteration: %f\n', min(accPSO_bestFitness(MaxIter, :), [], 2));
% fprintf('APSO Last Iteration Mean Value: %f\n', accPSO_meanFitness(end));
% fprintf('APSO Standard Deviation at Last Iteration: %f\n', accPSO_stddevFitness(end));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bestFitnessOverTime] = particleSwarmOptimization(Npop, nDimensions, maxIterations, fitnessFunction, LB, UB)
    cognitiveCoeff = 2;
    socialCoeff = 2;
    inertiaWeight = 0.5;

    positions = LB + (UB - LB) .* rand(Npop, nDimensions);
    velocities = zeros(Npop, nDimensions);

    fitness = zeros(Npop, 1);
    for i = 1:Npop
        fitness(i) = fitnessFunction(positions(i,:));
    end

    pBestPositions = positions;
    pBestFitness = fitness;
    [globalBestFitness, globalBestIndex] = min(fitness);
    gBestPosition = positions(globalBestIndex, :);

    bestFitnessOverTime = zeros(maxIterations, 1);

    for iteration = 1:maxIterations
        r1 = rand(Npop, nDimensions); 
        r2 = rand(Npop, nDimensions);

        velocities = inertiaWeight * velocities + ...
            cognitiveCoeff * r1 .* (pBestPositions - positions) + ...
            socialCoeff * r2 .* (gBestPosition - positions);

        positions = positions + velocities;
        positions = max(min(positions, UB), LB);

        fitness = zeros(Npop, 1);
        for i = 1:Npop
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
        bestFitnessOverTime(iteration) = globalBestFitness;
        fprintf('Iteration %d - PSO: Best Fitness = %f\n', iteration, globalBestFitness);
    end
end

function [bestFitnessOverTime] = acceleratedPSO(Npop, maxIterations, nDimensions, fitnessFunction, LB, UB, alpha, beta)
    inertiaWeight = 0.5;
    positions = LB + (UB - LB) .* rand(Npop, nDimensions);
    velocities = zeros(Npop, nDimensions);

    fitness = zeros(Npop, 1);
    for i = 1:Npop
        fitness(i) = fitnessFunction(positions(i,:));
    end

    pBestPositions = positions;
    pBestFitness = fitness;
    [globalBestFitness, globalBestIndex] = min(fitness);
    gBestPosition = positions(globalBestIndex, :);

    bestFitnessOverTime = zeros(maxIterations, 1);

    for iteration = 1:maxIterations
        epsilon = rand(Npop, nDimensions);
        velocities = velocities + alpha * (epsilon - 0.5) + beta * (gBestPosition - positions);
        positions = positions + velocities;
        positions = max(min(positions, UB), LB);

        fitness = zeros(Npop, 1);
        for i = 1:Npop
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
        bestFitnessOverTime(iteration) = globalBestFitness;
        fprintf('Iteration %d - APSO: Best Fitness = %f\n', iteration, globalBestFitness);
    end
end

function fitness_val = fitnessFunction(x)
    D = numel(x);
    sum_square = 0;
    sum_cos = 0;
    for i = 1:D
        sum_square = sum_square + x(i)^2;
        sum_cos = sum_cos + cos(2 * pi * x(i));
    end
    fitness_val = -20 * exp(-0.2 * sqrt(sum_square / D)) - exp(sum_cos / D) + 20 + exp(1);
end
