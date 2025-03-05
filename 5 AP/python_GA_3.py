import numpy as np
import matplotlib.pyplot as plt

class MaintenanceScheduler:
    def __init__(self):
        # Unit data: (capacity, maintenance_intervals_required)
        self.units = [
            (20, 2),  # Unit 1
            (15, 2),  # Unit 2
            (35, 1),  # Unit 3
            (40, 1),  # Unit 4
            (15, 1),  # Unit 5
            (15, 1),  # Unit 6
            (10, 1)   # Unit 7
        ]
        self.max_loads = [80, 90, 65, 70]  # MW per interval
        self.num_intervals = 4
        
        # GA parameters
        self.population_size = 50
        self.num_generations = 100
        self.crossover_prob  = 0.9
        self.mutation_prob = 0.01

    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = []
            for unit_idx, (_, maint_intervals) in enumerate(self.units):
                valid_schedules = self.get_valid_schedules(unit_idx)
                # Select a random schedule from valid options
                selected_schedule = valid_schedules[np.random.randint(len(valid_schedules))]
                chromosome.append(selected_schedule)
            population.append(chromosome)
        return np.array(population)


    def get_valid_schedules(self, unit_idx):
        maintenance_duration = self.units[unit_idx][1]
        if maintenance_duration == 2:
            return [[1,1,0,0], [0,1,1,0], [0,0,1,1]]
        else:  # maintenance_duration == 1
            return [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

    def calculate_fitness(self, chromosome):
        # Calculate available capacity per interval
        available_capacity = np.zeros(self.num_intervals)
        total_capacity = sum(unit[0] for unit in self.units)
        
        for interval in range(self.num_intervals):
            maintenance_units = []
            for unit_idx, (capacity, _) in enumerate(self.units):
                if chromosome[unit_idx][interval] == 1:
                    maintenance_units.append(capacity)
            
            available_capacity[interval] = total_capacity - sum(maintenance_units)
            
        # Calculate net reserve
        net_reserve = available_capacity - self.max_loads
        
        # Penalize negative reserves heavily
        if any(net_reserve < 0):
            return -1000
            
        # Objective is to maximize minimum reserve
        return min(net_reserve)

    def run(self):
        population = self.create_initial_population()
        best_fitness_history = []
        avg_fitness_history = []

        for generation in range(self.num_generations):
            # Evaluate fitness
            fitness_scores = [self.calculate_fitness(chrom) for chrom in population]
            
            # Store statistics
            best_fitness_history.append(max(fitness_scores))
            avg_fitness_history.append(np.mean(fitness_scores))
            
            # Selection
            selected_indices = self.selection(fitness_scores)
            population = population[selected_indices]
            
            # Crossover
            population = self.crossover(population)
            
            # Mutation
            population = self.mutation(population)
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {max(fitness_scores)}")

        self.plot_results(best_fitness_history, avg_fitness_history)
        return population[np.argmax(fitness_scores)]

    def selection(self, fitness_scores):
        fitness_scores = np.array(fitness_scores)
        fitness_scores = fitness_scores - min(fitness_scores) + 1e-6
        probs = fitness_scores / sum(fitness_scores)
        return np.random.choice(
            len(fitness_scores), 
            size=len(fitness_scores), 
            p=probs
        )

    def crossover(self, population):
        new_population = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                if np.random.random() < self.crossover_prob:
                    cross_point = np.random.randint(1, len(self.units))
                    child1 = np.concatenate([population[i][:cross_point], population[i+1][cross_point:]])
                    child2 = np.concatenate([population[i+1][:cross_point], population[i][cross_point:]])
                    new_population.extend([child1, child2])
                    
                else:
                    new_population.extend([population[i], population[i+1]])
        return np.array(new_population)

    def mutation(self, population):
        for i in range(len(population)):
            if np.random.random() < self.mutation_prob:
                unit_to_mutate = np.random.randint(0, len(self.units))
                valid_schedules = self.get_valid_schedules(unit_to_mutate)
                # Select a random valid schedule
                selected_schedule = valid_schedules[np.random.randint(len(valid_schedules))]
                population[i][unit_to_mutate] = selected_schedule
        return population


    def plot_results(self, best_fitness, avg_fitness):
        plt.figure(figsize=(10, 6))
        plt.plot(best_fitness, label='Best Fitness')
        plt.plot(avg_fitness, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Maintenance Scheduling Optimization Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_maintenance_schedule(self, best_schedule):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.arange(len(self.units))
        y = np.arange(self.num_intervals)
        X, Y = np.meshgrid(x, y)
        
        Z = np.array([schedule for schedule in best_schedule])
        
        ax.plot_surface(X, Y, Z.T, cmap='viridis')
        ax.set_xlabel('Unit Number')
        ax.set_ylabel('Time Interval')
        ax.set_zlabel('Maintenance Status')
        ax.set_title('Maintenance Schedule Visualization')
        plt.show()

    def plot_power_capacity(self, best_schedule):
        available_capacity = np.zeros(self.num_intervals)
        total_capacity = sum(unit[0] for unit in self.units)
        
        for interval in range(self.num_intervals):
            maintenance_units = []
            for unit_idx, (capacity, _) in enumerate(self.units):
                if best_schedule[unit_idx][interval] == 1:
                    maintenance_units.append(capacity)
            available_capacity[interval] = total_capacity - sum(maintenance_units)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.num_intervals), available_capacity, label='Available Capacity')
        plt.plot(range(self.num_intervals), self.max_loads, 'r-', label='Required Load', linewidth=2)
        plt.xlabel('Time Interval')
        plt.ylabel('Power (MW)')
        plt.title('Power Capacity vs Required Load')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    scheduler = MaintenanceScheduler()
    best_schedule = scheduler.run()
    print("Best maintenance schedule found:")
    print(best_schedule)
    scheduler.plot_maintenance_schedule(best_schedule)
    scheduler.plot_power_capacity(best_schedule)

