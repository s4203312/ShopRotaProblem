import collections
import pandas as pd 
import matplotlib.pyplot as plt
import randomcolor
import random

from ortools.sat.python import cp_model  #Import the CP-SAT solver.

#Read Data From CSV File
dataFile = pd.read_csv("rota_scheduling_dataset.csv", usecols=[0,1,2,3] ,header=None, skiprows=1)

assigned_task_type = tuple
all_tasks = tuple
all_machines = tuple
jobs_data = []
colours = []

def main():
    global assigned_task_type
    global all_tasks
    global all_machines
    global jobs_data

    #Creating a dataset from the csv file read by pandas
    jobs_data = DataSetCreation()
    #Creating the colour list so all jobs can be assigned a colour later
    ChartColourCreation()

    #Use genetic algorithm to try find best solution before using ORTools solvers
    #GeneticAlgorithm(model)            #Couldnt get it to work

    #Constraint Programming Solver
    model = ModelCreationCPSAT(jobs_data)
    SolveWithCpSat(model)

#Create required amount of colours for the charts
def ChartColourCreation():
    global colours

    grouped_tasks = dataFile.groupby(0)
    length_using_shape = grouped_tasks.size().shape[0]
    coloursRequired = length_using_shape        #Finding length of file to see how many colours are required
    i = 0
    while i < coloursRequired:
        #Generate a random color
        colours.append(randomcolor.RandomColor().generate())
        i += 1

#Getting information from the csv file and creating a useable dataset
def DataSetCreation():
    global jobs_data

    #Group tasks by job ID
    grouped_tasks = dataFile.groupby(0)

    #Iterate over groups and create jobs_data
    for _, job_tasks in grouped_tasks:
        tasks = []
        for _, task_row in job_tasks.iterrows():
            machine_id = task_row[2]  #Getting machine_id from the third column
            processing_time = task_row[3]  #Getting processing_time from the fourth column
            tasks.append((machine_id, processing_time))
        jobs_data.append(tasks)
    return jobs_data

#Create a model to use in the ORTools solvers
def ModelCreationCPSAT(jobs_data):
    global assigned_task_type
    global all_tasks
    global all_machines
    global horizon

    #Calculate the number of machines needed by finding the highest machine_id in jobs_data and adding 1.
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(               #Range is min inclusive, max exclusive
        machines_count
    )  #Create a range object for iterating over machine IDs.

    #Computes the planning horizon as the sum of all task durations to ensure enough time for all tasks.
    horizon = sum(task[1] for job in jobs_data for task in job)

    #Initialize the model.
    model = cp_model.CpModel()

    #Named tuple for easier task manipulation (holds the start, end, and interval variables for each task).
    #Tuples are fast and unchangable
    task_type = collections.namedtuple("task_type", "start end interval")

    #Named tuple for handling assigned tasks in the solution (includes start time, job and task indices, and duration).
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    #Stores task variables for all jobs and tasks.
    all_tasks = {}
    #Maps each machine to its corresponding tasks to enforce no overlap constraint.
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):        #Loop to find job and job_id
        for task_id, task in enumerate(job):        #Loop to find task and task_id
            machine, duration = task                #Each task has machine to run on and duration
            suffix = f"_{job_id}_{task_id}"         #Unique identifier for variables based on job and task ID.

            #Create variables for the start time, end time, and interval of each task.
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )

            #Store the task variables in a dictionary and add the interval variable to the machine's task list.
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    #Add disjunctive constraints to ensure that each machine processes at most one task at a time.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    #Add precedence constraints within each job to ensure tasks are performed in the specified order.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(
                all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
            )

    #Objective: Minimize the makespan (the total time to complete all jobs).
    obj_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
    )
    model.Minimize(obj_var)

    return model

#Use different solvers to compare results
def SolveWithCpSat(model):
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    name = "Constraint Programming"
    DisplaySolution(solver, status, name)

#Display the solutions with graphs
def DisplaySolution(solver, status, name):

    chart, axis = plt.subplots(figsize=(15, 6))
    chart.suptitle("Machine Tasks")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Solution: {name}")
        #Assign tasks to machines based on the solution.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1],
                    )                    
                )

        #Generate and print the schedule for each machine.
        output = ""
        for machine in all_machines:
            assigned_jobs[machine].sort()  #Sort tasks by start time.
            sol_line_tasks = "Machine " + str(machine + 1) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                axisjobnum = []
                name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                #Format the task information for printing.
                sol_line_tasks += f"{name:15}"
                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = f"[{start},{start + duration}]"
                sol_line += f"{sol_tmp:15}"
                #Add a bar to the graph
                bar = axis.barh(machine, width=duration, left=start, color=colours[assigned_task.job])
                axisjobnum.append(assigned_task.job)
                axis.bar_label(bar, labels=[f'Job:{num}' for num in axisjobnum], label_type='center')

            sol_line += "\n"
            sol_line_tasks += "\n"
            output += sol_line_tasks
            output += sol_line

        print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
        print(output)
        #Add y axis label and show graph
        axis.set_yticks(range(len(all_machines)))
        axis.set_yticklabels([f'{i+1}' for i in range(len(all_machines))])
        axis.set_ylabel('Machine')
        axis.invert_yaxis()
        axis.set_xlabel('Time(h)')
        plt.show()
    else:
        print("No solution found.")
    
    #Print statistics about the solution process.
    print("\nStatistics")
    print(f"  - conflicts: {solver.NumConflicts()}")
    print(f"  - branches : {solver.NumBranches()}")
    print(f"  - wall time: {solver.WallTime()}s")




#Trying to get genetic algorithm to work
def GeneticAlgorithm(model, population_size=100, max_generations=100, mutation_rate=0.1):
    current_population = []

    for _ in range(population_size):
        #Generate a random individual representing a schedule
        grouped_tasks = dataFile.groupby(0)
        num_jobs = grouped_tasks.size().shape[0]
        num_machines = 1 + max(task[0] for job in jobs_data for task in job)
        individual = generate_random_individual(num_jobs, num_machines)
        
        #Append the individual to the current population
        current_population.append(individual)

    for generation in range(max_generations):
        #Crossover
        parent1, parent2 = random.sample(current_population, 2)
        offspring = crossover(parent1, parent2)

        #Mutate
        if random.random() < mutation_rate:
            offspring = mutate(offspring)

        #Replace a randomly selected individual in the population with the offspring
        current_population[random.randint(0, population_size - 1)] = offspring
        
        print(f"Generation: {generation}")

    #Return the best solution found in the final population
    return max(current_population, key=fitness_function)

def generate_random_individual(num_jobs, num_machines):
    #Initialize an empty individual
    individual = []

    #Create a random schedule by assigning each job to a random machine
    for job_id in range(num_jobs):
        # Randomly select a machine ID for each job
        machine_id = random.randint(0, num_machines - 1)
        individual.append(machine_id)

    return individual

def crossover(parent1, parent2):
    #Create an empty offspring
    offspring = []

    #Determine the number of jobs in the parents
    num_jobs = min(len(parent1), len(parent2))

    #Perform one-point crossover
    crossover_point = random.randint(1, num_jobs - 1)

    #Combine parts of parent1 and parent2 to create the offspring
    offspring.extend(parent1[:crossover_point])
    offspring.extend(parent2[crossover_point:])

    return offspring

def mutate(offspring):
    global horizon
    mutated_offspring = offspring[:]  #Create a copy of the offspring to mutate

    #Select a random job to mutate
    job_id = random.randint(0, len(mutated_offspring) - 1)
    job = mutated_offspring[job_id]

    #Select a random task in the job to mutate its timing
    task_id = random.randint(0, len(job) - 1)
    task = job[task_id]

    #Mutate the start time of the selected task
    task[0] = random.randint(0, horizon)

    return mutated_offspring

def fitness_function(individual):
    total_processing_time = 0
    for job in individual:
        for task in job:
            total_processing_time += task[1]  #Add the processing time of each task
    return -total_processing_time  #Return negative total processing time (to minimize)


#Starts the program
if __name__ == "__main__":
    main()