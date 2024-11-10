% clear;
% close all;
% 
% centroids = 2;
% dimensions = 2;
% particles = 20;
% iterations = 50;
% dataset_subset = 2;
% load fisheriris.mat
% meas = meas(:,1+dataset_subset:dimensions+dataset_subset);
% 
% w  = 0.72;
% c1 = 1.49;
% c2 = 1.49;
% 
% fh = figure(1);
% hold on;
% if dimensions == 3
%     plot3(meas(:,1),meas(:,2),meas(:,3),'k*');
%     view(3);
% elseif dimensions == 2
%     plot(meas(:,1),meas(:,2),'k*');
% end
% 
% axis equal;
% axis(reshape([min(meas)-2; max(meas)+2],1,[]));
% hold off;
% 
% swarm_vel = rand(centroids,dimensions,particles)*0.1;
% swarm_pos = rand(centroids,dimensions,particles);
% swarm_best = zeros(centroids,dimensions);
% c = zeros(size(meas,1),particles);
% ranges = max(meas)-min(meas); 
% swarm_pos = swarm_pos .* repmat(ranges,centroids,1,particles) + repmat(min(meas),centroids,1,particles);
% swarm_fitness = Inf(1,particles);
% 
% for iteration = 1:iterations
%     distances = zeros(size(meas,1),centroids,particles);
% 
%     for particle = 1:particles
%         for centroid = 1:centroids
%             for data_vector = 1:size(meas,1)
%                 distances(data_vector,centroid,particle) = norm(swarm_pos(centroid,:,particle) - meas(data_vector,:));
%             end
%         end
%     end
% 
%     for particle = 1:particles
%         [~, index] = min(distances(:,:,particle),[],2);
%         c(:,particle) = index;
%     end
% 
%     average_fitness = zeros(particles,1);
%     for particle = 1:particles
%         for centroid = 1:centroids
%             if any(c(:,particle) == centroid)
%                 local_fitness = mean(distances(c(:,particle)==centroid,centroid,particle));
%                 average_fitness(particle) = average_fitness(particle) + local_fitness;
%             end
%         end
%         average_fitness(particle) = average_fitness(particle) / centroids;
%         if (average_fitness(particle) < swarm_fitness(particle))
%             swarm_fitness(particle) = average_fitness(particle);
%             swarm_best(:,:,particle) = swarm_pos(:,:,particle); 
%         end
%     end    
%     [global_fitness, index] = min(swarm_fitness);       
%     swarm_overall_pose = swarm_pos(:,:,index);         
% 
%     r1 = rand;
%     r2 = rand;
%     for particle = 1:particles        
%         inertia = w * swarm_vel(:,:,particle);
%         cognitive = c1 * r1 * (swarm_best(:,:,particle) - swarm_pos(:,:,particle));
%         social = c2 * r2 * (swarm_overall_pose - swarm_pos(:,:,particle));
%         vel = inertia + cognitive + social;
% 
%         swarm_pos(:,:,particle) = swarm_pos(:,:,particle) + vel;   
%         swarm_vel(:,:,particle) = vel; 
%     end
% end
% 
% hold on;
% particle = index; 
% cluster_colors = ['m','g','y','b','r','c','g'];
% for centroid = 1:centroids
%     if any(c(:,particle) == centroid)
%         if dimensions == 3
%             plot3(meas(c(:,particle)==centroid,1),meas(c(:,particle)==centroid,2),meas(c(:,particle)==centroid,3),'o','color',cluster_colors(centroid));
%         elseif dimensions == 2
%             plot(meas(c(:,particle)==centroid,1),meas(c(:,particle)==centroid,2),'o','color',cluster_colors(centroid));
%         end
%     end
% end
% hold off;
% 
% fprintf('\nEnd, global fitness is %5.4f\n',global_fitness);

clc;
clear;
data = readtable('/MATLAB Drive/Dataset/Mall_Customers.csv', 'VariableNamingRule', 'preserve');
data.Gender = grp2idx(data.Gender) - 1;
processed_data = data{:, {'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'}};
processed_data = (processed_data - mean(processed_data)) ./ std(processed_data);

[coeff, score, ~, ~, explained] = pca(processed_data);

for i = 1:length(explained)
    fprintf('Component %d: %.2f%%\n', i, explained(i));
end

pca_data = score(:, 1:2);

centroids = 3;
dimensions = size(pca_data, 2);
particles = 20;
iterations = 50;
w = 0.72;
c1 = 1.49;
c2 = 1.49;

ranges = max(pca_data) - min(pca_data);
swarm_pos = rand(centroids, dimensions, particles) .* reshape(ranges, [1, dimensions, 1]) + min(pca_data);
swarm_vel = rand(centroids, dimensions, particles) * 0.1;

swarm_fitness = inf(1, particles);
swarm_best = zeros(centroids, dimensions, particles);
dataset_size = size(pca_data, 1);
c = zeros(dataset_size, particles);

for iteration = 1:iterations
    distances = zeros(dataset_size, centroids, particles);
    for particle = 1:particles
        for centroid = 1:centroids
            for data_point = 1:dataset_size
                distances(data_point, centroid, particle) = sqrt(sum((pca_data(data_point, :) - swarm_pos(centroid, :, particle)).^2));
            end
        end
    end

    for particle = 1:particles
        for data_point = 1:dataset_size
            [~, c(data_point, particle)] = min(distances(data_point, :, particle));
        end
    end

    average_fitness = zeros(1, particles);
    for particle = 1:particles
        for centroid = 1:centroids
            local_fitness = 0;
            for data_point = 1:dataset_size
                if c(data_point, particle) == centroid
                    local_fitness = local_fitness + distances(data_point, centroid, particle);
                end
            end
            average_fitness(particle) = average_fitness(particle) + local_fitness / sum(c(:, particle) == centroid);
        end
        if average_fitness(particle) < swarm_fitness(particle)
            swarm_fitness(particle) = average_fitness(particle);
            swarm_best(:, :, particle) = swarm_pos(:, :, particle);
        end
    end

    [global_best_fitness, best_particle] = min(swarm_fitness);
    global_best_position = swarm_pos(:, :, best_particle);

    r1 = rand;
    r2 = rand;
    for particle = 1:particles
        inertia = w * swarm_vel(:, :, particle);
        cognitive = c1 * r1 * (swarm_best(:, :, particle) - swarm_pos(:, :, particle));
        social = c2 * r2 * (global_best_position - swarm_pos(:, :, particle));
        swarm_vel(:, :, particle) = inertia + cognitive + social;
        swarm_pos(:, :, particle) = swarm_pos(:, :, particle) + swarm_vel(:, :, particle);
    end

    fprintf('Iteration %d, Global Fitness: %.4f\n', iteration, global_best_fitness);
end

fprintf('Final Global Fitness (Quantization Error): %.4f\n', global_best_fitness);

figure;
hold on;
colors = ['r', 'g', 'b'];
for centroid = 1:centroids
    cluster_points = pca_data(c(:, best_particle) == centroid, :);
    scatter(cluster_points(:, 1), cluster_points(:, 2), [], colors(centroid), 'filled');
end
title('PSO Clustering Results After PCA');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
hold off;

