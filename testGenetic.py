import random
import collections  # Provides access to specialized container datatypes.
import pandas as pd 
import matplotlib.pyplot as plt

#Read Data From CSV File
dataFile = pd.read_csv("rota_scheduling_dataset.csv", usecols=[0,1,2,3] ,header=None, skiprows=1)

# Genetic Algorithm Parameters
population_size = 60
mutation_rate = 0.1
number_generations = 40

def main():
    global number_jobs
    global number_machines
    global best_individual
    
    # Set global variables
    number_jobs, number_machines, jobs_data = read_data_from_file()
    
    # Run genetic algorithm
    population = initialize_population(number_jobs)
    #Loop for number of generations
    for generation in range(number_generations):
        #Update the population
        population = evolve_population(population, jobs_data)

        print(f"Generation {generation + 1}, Best Makespan: {fitness_function(best_individual, jobs_data)}")
    
    # Assuming colours and all_machines are defined elsewhere in your code
    colours = ['blue', 'green', 'red', 'yellow', 'orange','blue', 'green', 'red', 'yellow', 'orange','blue', 'green', 'red', 'yellow', 'orange','blue', 'green', 'red', 'yellow', 'orange','blue', 'green', 'red', 'yellow', 'orange','blue', 'green', 'red', 'yellow', 'orange','blue', 'green', 'red', 'yellow', 'orange','blue', 'green', 'red', 'yellow', 'orange',]  # Define your own colours
    all_machines = range(number_machines)  # Assuming you have the number_machines defined  

    # Now you can call DisplaySolution with the population, jobs_data, and the index of the best individual
    best_individual_index = population.index(min(population, key=lambda x: fitness_function(x, jobs_data)))
    DisplaySolution(population, jobs_data, best_individual_index, all_machines, colours)
    
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
    number_jobs = grouped_tasks.size().shape[0]
    print(number_jobs)
    number_machines = 1 + max(task[0] for job in jobs_data for task in job)
    print(number_machines)

    
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
            start_time = max(machine_timings[machine_id], machine_timings[machine_id-1]) + processing_time
            machine_timings[machine_id] = start_time
        machine_id -= 1
    return max(machine_timings)

def evolve_population(population, jobs_data):
    global best_individual
    new_population = []

    # Find the best result so far (elitism)
    best_individual = min(population, key=lambda x: fitness_function(x, jobs_data))
    new_population.append(best_individual)

    # Generate offspring until the new population is full
    while len(new_population) < len(population):
        # Tournament selection
        parents = random.sample(population, 2)
        parent1, parent2 = min(parents, key=lambda x: fitness_function(x, jobs_data)), max(parents, key=lambda x: fitness_function(x, jobs_data))

        # Apply crossover
        child = crossover(parent1, parent2)

        # Apply mutation
        mutate(child)

        new_population.append(child)

    return new_population





def DisplaySolution(population, jobs_data, best_individual_index, all_machines, colours):
    # Assuming jobs_data and all_machines are defined globally or passed as arguments

    best_individual = population[best_individual_index]

    chart, axis = plt.subplots(figsize=(15, 6))
    chart.suptitle("Machine Tasks")

    print("Solution:")
    # Assign tasks to machines based on the solution.
    assigned_jobs = collections.defaultdict(list)
    for job_id in best_individual:
        job = jobs_data[job_id]
        for task_id, task in enumerate(job):
            machine = task[0]
            assigned_jobs[machine].append(
                assigned_task_type(
                    start=start_time_for_task(job_id, task_id, best_individual, jobs_data),
                    job=job_id,
                    index=task_id,
                    duration=task[1],
                )                    
            )

    # Generate and print the schedule for each machine.
    output = ""
    for machine in all_machines:
        assigned_jobs[machine].sort()  # Sort tasks by start time.
        sol_line_tasks = "Machine " + str(machine + 1) + ": "
        sol_line = "           "
        axisjobnum = []

        for assigned_task in assigned_jobs[machine]:
            name = f"job_{assigned_task.job}_task_{assigned_task.index}"
            # Format the task information for printing.
            sol_line_tasks += f"{name:15}"
            start = assigned_task.start
            duration = assigned_task.duration
            sol_tmp = f"[{start},{start + duration}]"
            sol_line += f"{sol_tmp:15}"
            # Add a bar to the graph
            bar = axis.barh(machine, width=duration, left=start, color=colours[assigned_task.job])
            axisjobnum.append(assigned_task.job)
            axis.bar_label(bar, labels=[f'Job:{num}' for num in axisjobnum], label_type='center')

        sol_line += "\n"
        sol_line_tasks += "\n"
        output += sol_line_tasks
        output += sol_line

    print(output)
    
    # Show the plot
    plt.xlabel('Time(h)')
    plt.ylabel('Machine')
    plt.yticks(range(len(all_machines)), [f'Machine {i+1}' for i in range(len(all_machines))])
    plt.gca().invert_yaxis()
    plt.show()

# Define assigned_task_type
assigned_task_type = collections.namedtuple('AssignedTask', ['start', 'job', 'index', 'duration'])

def start_time_for_task(job_id, task_id, schedule, jobs_data):
    start_time = 0
    for j in range(job_id):
        for task in jobs_data[schedule[j]]:
            start_time += task[1]
    current_job = jobs_data[schedule[job_id]]
    print("Current Job:", current_job)  # Print current job
    for t in range(task_id):
        try:
            start_time += current_job[t][1]
        except IndexError:
            print("IndexError occurred at job_id:", job_id, "task_id:", task_id, "schedule:", schedule)
    return start_time


if __name__ == '__main__':
    main()
