import random
import math
import pandas as pd 

#Read Data From CSV File
dataFile = pd.read_csv("rota_scheduling_dataset.csv", usecols=[0,1,2,3] ,header=None, skiprows=1)

# Genetic Algorithm Parameters
population_size = 40
mutation_rate = 0.01
number_generations = 40

def main():
    global number_jobs
    global number_machines
    
    # Set global variables
    number_jobs, number_machines, jobs_data = read_data_from_file()
    
    # Run genetic algorithm
    population = initialize_population(number_jobs)
    for generation in range(number_generations):
        population = evolve_population(population, jobs_data)
        best_individual = min(population, key=lambda x: fitness_function(x, jobs_data))
        print(f"Generation {generation + 1}, Best Makespan: {fitness_function(best_individual, jobs_data)}")
    
def read_data_from_file():

    jobs_data = []
    # Group tasks by job ID
    grouped_tasks = dataFile.groupby(0)

    # Iterate over groups and create jobs_data
    for _, job_tasks in grouped_tasks:
        tasks = []
        for _, task_row in job_tasks.iterrows():
            machine_id = task_row[2]  # Getting machine_id from the third column
            processing_time = task_row[3]  # Getting processing_time from the fourth column
            tasks.append((machine_id, processing_time))
        jobs_data.append(tasks)

    print(jobs_data)

    # Read data from file or define it here
    # For simplicity, let's assume data is predefined
    number_jobs = 40
    number_machines = 15
    
    return number_jobs, number_machines, jobs_data

def initialize_population(number_jobs):
    population = []
    for _ in range(population_size):
        population.append(random.sample(range(number_jobs), number_jobs))
    return population

def crossover(parent1, parent2):
    # Implement crossover operator (e.g., partially mapped crossover)
    crossover_point = random.randint(0, len(parent1))
    child = parent1[:crossover_point]
    for job in parent2:
        if job not in child:
            child.append(job)
    return child

def mutate(schedule):
    if random.random() < mutation_rate:
        # Implement mutation operator (e.g., swap mutation)
        idx1, idx2 = random.sample(range(len(schedule)), 2)
        schedule[idx1], schedule[idx2] = schedule[idx2], schedule[idx1]

def fitness_function(schedule, jobs_data):
    machine_timings = [0] * number_machines
    for job_id in schedule:
        job = jobs_data[job_id]
        for task in job:
            machine_id, processing_time = task
            print(f"Machine ID: {machine_id}, Number of Machines: {number_machines}")
            start_time = max(machine_timings[machine_id], machine_timings[machine_id]) + processing_time
            machine_timings[machine_id] = start_time
    return max(machine_timings)

def evolve_population(population, jobs_data):
    # Select parents
    parents = random.sample(population, 2)
    # Apply crossover
    child = crossover(parents[0], parents[1])
    # Apply mutation
    mutate(child)
    # Replace a random individual in the population with the child
    population[random.randint(0, len(population) - 1)] = child
    return population

if __name__ == '__main__':
    main()
