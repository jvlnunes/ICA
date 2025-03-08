classdef MaintenanceScheduler
    properties
        units
        max_loads
        num_intervals
        population_size
        num_generations
        crossover_prob
        mutation_prob
    end
    
    methods
        function obj = MaintenanceScheduler()
            % Unit data: (capacity, maintenance_intervals_required)
            obj.units = [
                20, 2;  % Unit 1
                15, 2;  % Unit 2
                35, 1;  % Unit 3
                40, 1;  % Unit 4
                15, 1;  % Unit 5
                15, 1;  % Unit 6
                10, 1   % Unit 7
            ];
            obj.max_loads = [80, 90, 65, 70];  % MW per interval
            obj.num_intervals = 4;

            % GA parameters
            obj.population_size = 50;
            obj.num_generations = 100;
            obj.crossover_prob = 0.9;
            obj.mutation_prob = 0.01;
        end

        function population = create_initial_population(obj)
            population = cell(obj.population_size, 1);
            for i = 1:obj.population_size
                chromosome = cell(size(obj.units, 1), 1);
                for unit_idx = 1:size(obj.units, 1)
                    valid_schedules = obj.get_valid_schedules(unit_idx);
                    selected_schedule = valid_schedules(randi(size(valid_schedules, 1)), :);
                    chromosome{unit_idx} = selected_schedule;
                end
                population{i} = chromosome;
            end
        end

        function valid_schedules = get_valid_schedules(obj, unit_idx)
            maintenance_duration = obj.units(unit_idx, 2);
            if maintenance_duration == 2
                valid_schedules = [
                    1, 1, 0, 0;
                    0, 1, 1, 0;
                    0, 0, 1, 1
                ];
            else
                valid_schedules = [
                    1, 0, 0, 0;
                    0, 1, 0, 0;
                    0, 0, 1, 0;
                    0, 0, 0, 1
                ];
            end
        end

        function fitness = calculate_fitness(obj, chromosome)
            available_capacity = zeros(1, obj.num_intervals);
            total_capacity = sum(obj.units(:, 1));
            
            for interval = 1:obj.num_intervals
                maintenance_units = [];
                for unit_idx = 1:size(obj.units, 1)
                    if chromosome{unit_idx}(interval) == 1
                        maintenance_units = [maintenance_units; obj.units(unit_idx, 1)];
                    end
                end
                available_capacity(interval) = total_capacity - sum(maintenance_units);
            end
            
            net_reserve = available_capacity - obj.max_loads;
            if any(net_reserve < 0)
                fitness = -1000;
            else
                fitness = min(net_reserve);
            end
        end

        function best_schedule = run(obj)
            population = obj.create_initial_population();
            best_fitness_history = [];
            avg_fitness_history = [];

            for generation = 1:obj.num_generations
                fitness_scores = cellfun(@(chrom) obj.calculate_fitness(chrom), population);
                best_fitness_history = [best_fitness_history; max(fitness_scores)];
                avg_fitness_history = [avg_fitness_history; mean(fitness_scores)];

                selected_indices = obj.selection(fitness_scores);
                population = population(selected_indices);

                population = obj.crossover(population);
                population = obj.mutation(population);

                if mod(generation, 10) == 0
                    fprintf('Generation %d: Best fitness = %f\n', generation, max(fitness_scores));
                end
            end

            [~, best_index] = max(fitness_scores);
            best_schedule = population{best_index};
            obj.plot_results(best_fitness_history, avg_fitness_history);
        end

        function selected_indices = selection(obj, fitness_scores)
            fitness_scores = fitness_scores - min(fitness_scores) + 1e-6;
            probs = fitness_scores / sum(fitness_scores);
            selected_indices = randsample(1:length(fitness_scores), length(fitness_scores), true, probs);
        end

        function new_population = crossover(obj, population)
            new_population = {};
            for i = 1:2:length(population)
                if i + 1 <= length(population)
                    if rand < obj.crossover_prob
                        cross_point = randi([1, size(obj.units, 1)]);
                        child1 = [population{i}(1:cross_point); population{i+1}(cross_point+1:end)];
                        child2 = [population{i+1}(1:cross_point); population{i}(cross_point+1:end)];
                        new_population{end+1} = child1;
                        new_population{end+1} = child2;
                    else
                        new_population{end+1} = population{i};
                        new_population{end+1} = population{i+1};
                    end
                end
            end
        end

        function population = mutation(obj, population)
            for i = 1:length(population)
                if rand < obj.mutation_prob
                    unit_to_mutate = randi(size(obj.units, 1));
                    valid_schedules = obj.get_valid_schedules(unit_to_mutate);
                    selected_schedule = valid_schedules(randi(size(valid_schedules, 1)), :);
                    population{i}{unit_to_mutate} = selected_schedule;
                end
            end
        end

        function plot_results(~, best_fitness, avg_fitness)
            figure;
            plot(best_fitness, 'DisplayName', 'Best Fitness');
            hold on;
            plot(avg_fitness, 'DisplayName', 'Average Fitness');
            xlabel('Generation');
            ylabel('Fitness');
            title('Maintenance Scheduling Optimization Progress');
            legend;
            grid on;
            hold off;
        end
    end
end

% Main execution
scheduler = MaintenanceScheduler();
best_schedule = scheduler.run();
disp('Best maintenance schedule found:');
disp(best_schedule);
