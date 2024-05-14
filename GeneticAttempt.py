import random
import collections
import pandas as pd 
import matplotlib.pyplot as plt

#Read Data From CSV File
dataFile = pd.read_csv("rota_scheduling_dataset.csv", usecols=[0,1,2,3] ,header=None, skiprows=1)

population_size = 100
mutation_rate = 0.1
number_generations = 60

def main():
    global number_jobs
    global number_machines
    global best_individual
    
    #Set global variables
    number_jobs, number_machines, jobs_data = read_data_from_file()
    print(number_machines)
    #Create initial population
    population = initialize_population(number_jobs)

    #Loop for number of generations
    for generation in range(number_generations):
        #Update the population
        population = evolve_population(population, jobs_data)

        print(f"Generation {generation + 1}, Best Makespan: {fitness_function(best_individual, jobs_data)}")
    
    #Displaying the solution
    print(f"Final Best Solution: {best_individual}")
    
def read_data_from_file():

    jobs_data = []
    grouped_tasks = dataFile.groupby(0)

    #Create jobs_data
    for _, job_tasks in grouped_tasks:
        tasks = []
        for _, task_row in job_tasks.iterrows():
            machine_id = task_row[2]  #Getting machine_id from the third column
            processing_time = task_row[3]  #Getting processing_time from the fourth column
            tasks.append((machine_id, processing_time))
        jobs_data.append(tasks)

    print(jobs_data)

    number_jobs = grouped_tasks.size().shape[0]
    number_machines = 1 + max(task[0] for job in jobs_data for task in job)

    return number_jobs, number_machines, jobs_data

def initialize_population(number_jobs):
    population = []
    for _ in range(population_size):
        population.append(random.sample(range(number_jobs), number_jobs))   #Gets random sample from the number of jobs
    return population

def crossover(parent1, parent2):
    #Perform crossover functionality
    crossover_point = random.randint(0, len(parent1))
    child = parent1[:crossover_point]
    for job in parent2:
        if job not in child:
            child.append(job)
    return child

def mutate(schedule):
    if random.random() < mutation_rate:
        #Perform mutation functionality
        sample1, sample2 = random.sample(range(len(schedule)), 2)
        schedule[sample1], schedule[sample2] = schedule[sample2], schedule[sample1]

def fitness_function(schedule, jobs_data):
    machine_timings = [0] * number_machines
    for job_id in schedule:
        job = jobs_data[job_id]
        for task in job:
            machine_id, processing_time = task
            start_time = max(machine_timings[machine_id], machine_timings[machine_id-1]) + processing_time
            machine_timings[machine_id] = start_time
        machine_id -= 1
    return max(machine_timings)

def evolve_population(population, jobs_data):
    global best_individual
    new_population = []

    #Find the best result so far (elitism)
    best_individual = min(population, key=lambda x: fitness_function(x, jobs_data))
    new_population.append(best_individual)

    #Refill the population
    while len(new_population) < len(population):
        #Find two random parents
        parents = random.sample(population, 2)
        parent1, parent2 = min(parents, key=lambda x: fitness_function(x, jobs_data)), max(parents, key=lambda x: fitness_function(x, jobs_data))

        #Crossover
        child = crossover(parent1, parent2)
        #Mutation
        mutate(child)

        new_population.append(child)

    return new_population

if __name__ == '__main__':
    main()
