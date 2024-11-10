function [bestFitnessOverTime] = particleSwarmOptimization(nParticles, maxIterations, nDimensions, LB, UB, chaoticFunc)
    cognitiveCoeff = 2; 
    socialCoeff = 2;    
    inertiaWeight = 0.5; 

    positions = zeros(nParticles, nDimensions);  
    velocities = zeros(nParticles, nDimensions);  
    
    for i = 1:nParticles
        positions(i, :) = chaoticFunc(nDimensions);  
        velocities(i, :) = chaoticFunc(nDimensions);  
    end

    for i = 1:nParticles
        positions(i, :) = LB + (UB - LB) .* (positions(i, :) - min(positions(i, :))) / (max(positions(i, :)) - min(positions(i, :)));
    end

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
        r1 = zeros(nParticles, nDimensions);  
        r2 = zeros(nParticles, nDimensions);  
        
        for i = 1:nParticles
            r1(i, :) = chaoticFunc(nDimensions);  
            r2(i, :) = chaoticFunc(nDimensions);  
        end

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%20 Runs Main 
MaxIter = 1000;
D = 6;
Npop = 50;
nRuns = 25;

LB = -10 * ones(1, D);
UB = 10 * ones(1, D);

PSO_bestFitness = zeros(MaxIter, 6);
PSO_meanFitness = zeros(MaxIter, 6);
PSO_stdFitness = zeros(1, 6);
PSO_globalBest = zeros(1, 6);
PSO_meanLast = zeros(1, 6);
PSO_stdLast = zeros(1, 6);
PSO_globalBestLast = zeros(1, 6);

chaoticMaps = {@LogisticMap, @CircleMap, @KentMap, @PiecewiseMap, @SineMap, @SinosoidalMap};
mapNames = {'LogisticMap', 'CircleMap', 'KentMap', 'PiecewiseMap', 'SineMap', 'SinosoidalMap'};

for mapIdx = 1:length(chaoticMaps)
    chaoticMapFunction = chaoticMaps{mapIdx};
    bestFitnessOverAllRuns = zeros(MaxIter, nRuns);

    for run = 1:nRuns
        bestFitnessOverTime = particleSwarmOptimization(Npop, MaxIter, D, LB, UB, chaoticMapFunction);
        bestFitnessOverAllRuns(:, run) = bestFitnessOverTime;
    end

    PSO_bestFitness(:, mapIdx) = min(bestFitnessOverAllRuns, [], 2);
    PSO_meanFitness(:, mapIdx) = mean(bestFitnessOverAllRuns, 2);
    PSO_stdFitness(mapIdx) = std(bestFitnessOverAllRuns(:));

    PSO_globalBest(mapIdx) = min(PSO_bestFitness(:, mapIdx));

    PSO_meanLast(mapIdx) = PSO_meanFitness(MaxIter, mapIdx);
    PSO_stdLast(mapIdx) = PSO_stdFitness(mapIdx);
    PSO_globalBestLast(mapIdx) = PSO_bestFitness(MaxIter, mapIdx);

    fprintf('Standard Deviation of Global Best Fitness for %s: %.4f\n', func2str(chaoticMaps{mapIdx}), PSO_stdFitness(mapIdx));
end

T = table(mapNames', PSO_stdLast', PSO_meanLast', PSO_globalBestLast', ...
    'VariableNames', {'Map', 'StandardDeviation', 'MeanFitness', 'GlobalBestFitness'});

disp(T);

figure;
hold on;
colors = lines(length(chaoticMaps));
for i = 1:length(chaoticMaps)
    plot(1:MaxIter, PSO_bestFitness(:, i), 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', mapNames{i});
end
xlabel('Iteration');
ylabel('Best Fitness Value');
title('Best Fitness Value Over Iterations for Different Chaotic Maps');
legend('show');
grid on;

figure;
hold on;
for i = 1:length(chaoticMaps)
    plot(1:MaxIter, PSO_meanFitness(:, i), 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', mapNames{i});
end
xlabel('Iteration');
ylabel('Mean Fitness Value');
title('Mean Fitness Value Over Iterations for Different Chaotic Maps');
legend('show');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Single Run Main
% MaxIter = 1000;
% D = 6;
% Npop = 50;
% 
% LB = -10 * ones(1, D);
% UB = 10 * ones(1, D);
% 
% PSO_bestFitness = zeros(MaxIter, 6);
% PSO_meanFitness = zeros(MaxIter, 6);
% 
% chaoticMaps = {@LogisticMap, @CircleMap, @KentMap, @PiecewiseMap, @SineMap, @SinosoidalMap};
% mapNames = {'LogisticMap', 'CircleMap', 'KentMap', 'PiecewiseMap', 'SineMap', 'SinosoidalMap'};
% 
% figure;  
% hold on;  
% 
% for mapIdx = 1:length(chaoticMaps)
%     chaoticMapFunction = chaoticMaps{mapIdx};
%     bestFitnessOverTime = particleSwarmOptimization(Npop, MaxIter, D, LB, UB, chaoticMapFunction);
% 
%     PSO_bestFitness(:, mapIdx) = bestFitnessOverTime;
%     PSO_meanFitness(:, mapIdx) = mean(PSO_bestFitness, 2);
% 
%     bestValue = PSO_bestFitness(end, mapIdx);  
% 
%     fprintf('Convergence plot for %s\n', mapNames{mapIdx});
%     fprintf('Best fitness value for %s: %f\n', mapNames{mapIdx}, bestValue);
% 
%     plot(1:MaxIter, PSO_bestFitness(:, mapIdx), 'LineWidth', 2, 'DisplayName', mapNames{mapIdx});
% end
% 
% xlabel('Iteration');
% ylabel('Best Fitness Value');
% title('Best Fitness Value Over Iterations for Different Chaotic Maps');
% legend('show');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Cha] = LogisticMap(n)
    x = ones(1, n);
    x(1) = rand;
    A = [0.00 0.25 0.50 0.75 1.00];
    while ismember(x(1), A)
        x(1) = rand;
    end
    for i = 1:n-1
        x(i+1) = 4 * x(i) * (1 - x(i));
    end
    Cha = x;
end

function [Cha] = CircleMap(n)
    x = ones(1, n);
    x(1) = rand;
    for i = 1:n-1
        x(i+1) = mod(x(i) + 0.1, 1);
    end
    Cha = x;
end

function [Cha] = KentMap(n)
    x = ones(1, n);
    m = rand;
    x(1) = rand;
    for i = 1:n-1
        if x(i) <= m
            x(i+1) = x(i) / m;
        else
            x(i+1) = (1 - x(i)) / (1 - m);
        end
    end
    Cha = x;
end

function [Cha] = PiecewiseMap(n)
    x = ones(1, n);
    x(1) = rand;
    for i = 1:n-1
        if x(i) <= 0.7
            x(i+1) = x(i) / 0.7;
        else
            x(i+1) = (1 - x(i)) / (1 - 0.7);
        end
    end
    Cha = x;
end

function [Cha] = SineMap(n)
    x = ones(1, n);
    x(1) = rand;
    for i = 1:n-1
        x(i+1) = sin(pi * x(i));
    end
    Cha = x;
end

function [Cha] = SinosoidalMap(n)
    x = ones(1, n);
    x(1) = 0.55;
    for i = 1:n-1
        x(i+1) = 2.3 * x(i)^2 * sin(pi * x(i));
    end
    Cha = x;
end