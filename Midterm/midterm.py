import numpy as np
import matplotlib.pyplot as plt

# Parameters
population_size = 10
genome_length = 20  # N = 20
num_generations = 100
mutation_rate = 0.01

# Target genome (N ones)
target_genome = np.ones(genome_length)

# Function to create initial population
def initialize_population(size, genome_length):
    return np.random.randint(2, size=(size, genome_length))

# Fitness function
def fitness(individual):
    return np.sum(individual == target_genome)

# Selection (tournament selection)
def select_parents(population):
    tournament_size = 3
    selected = []
    for _ in range(population_size):
        # Select random indices for the tournament
        indices = np.random.choice(len(population), tournament_size)
        tournament = population[indices]
        winner = tournament[np.argmax([fitness(ind) for ind in tournament])]
        selected.append(winner)
    return np.array(selected)

# Crossover function
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, genome_length - 1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

# Mutation function
def mutate(individual):
    for i in range(genome_length):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip the bit
    return individual

# Main evolutionary algorithm
population = initialize_population(population_size, genome_length)
max_fitness_values = []

for generation in range(num_generations):
    fitness_values = [fitness(ind) for ind in population]
    max_fitness = max(fitness_values)
    max_fitness_values.append(max_fitness)

    print(f'Generation {generation}: Max Fitness = {max_fitness}')

    # Selection
    selected_parents = select_parents(population)
    
    # Create the next generation
    next_generation = []
    for i in range(0, population_size, 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent1, parent2)
        next_generation.append(mutate(child1))
        next_generation.append(mutate(child2))
    
    population = np.array(next_generation)

# Plotting
plt.plot(max_fitness_values)
plt.title('Max Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Max Fitness')
plt.grid()
plt.xticks(range(0, num_generations, 5))  # Show every 5 generations on the x-axis
plt.show()
plt.savefig('my_plot3.png', dpi=300)
