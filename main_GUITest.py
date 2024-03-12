#Include own dataset form better grade****** done ** need to find data to fill out set
#Mabye try different solvers******
#Create a visual representation of the output?******

import collections  # Provides access to specialized container datatypes.
import customtkinter as ctk
import pandas as pd 
import matplotlib.pyplot as plt

from ortools.sat.python import cp_model  # Import the CP-SAT solver.

#Read Data From CSV File
dataFile = pd.read_csv("OptimizationDataSet.csv", usecols=[0,1,2,3] ,header=None, skiprows=1)

assigned_task_type = tuple
all_tasks = tuple
all_machines = tuple
jobs_data = []
colours = [("Black"), ("Blue"), ("Red"), ("Yellow"), ("Green")]

def main():
    global assigned_task_type
    global all_tasks
    global all_machines
    global jobs_data

    #Creating a dataset from the csv file read by pandas
    jobs_data = DataSetCreation()

    #Create model for ORtools use
    model = ModelCreation(jobs_data)

    # Solve the model using the CP-SAT solver.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    #Displaying Result
    DisplaySolution(solver, status)
    


def DataSetCreation():
    global jobs_data

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
    return jobs_data

def ModelCreation(jobs_data):
    global assigned_task_type
    global all_tasks
    global all_machines

# Calculate the number of machines needed by finding the highest machine_id in jobs_data and adding 1.
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(               #Range is min inclusive, max exclusive
        machines_count
    )  # Create a range object for iterating over machine IDs.

    # Computes the planning horizon as the sum of all task durations to ensure enough time for all tasks.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Initialize the model.
    model = cp_model.CpModel()

    # Named tuple for easier task manipulation (holds the start, end, and interval variables for each task).
    #Tuples are fast and unchangable
    task_type = collections.namedtuple("task_type", "start end interval")

    # Named tuple for handling assigned tasks in the solution (includes start time, job and task indices, and duration).
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    # Stores task variables for all jobs and tasks.
    all_tasks = {}
    # Maps each machine to its corresponding tasks to enforce no overlap constraint.
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):        #Loop to find job and job_id
        for task_id, task in enumerate(job):        #Loop to find task and task_id
            machine, duration = task                #Each task has machine to run on and duration
            suffix = f"_{job_id}_{task_id}"  # Unique identifier for variables based on job and task ID.

            # Create variables for the start time, end time, and interval of each task.
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )

            # Store the task variables in a dictionary and add the interval variable to the machine's task list.
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    # Add disjunctive constraints to ensure that each machine processes at most one task at a time.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Add precedence constraints within each job to ensure tasks are performed in the specified order.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(
                all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
            )

    # Objective: Minimize the makespan (the total time to complete all jobs).
    obj_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
    )
    model.Minimize(obj_var)

    return model

def DisplaySolution(solver, status):

    chart, axis = plt.subplots()
    chart.suptitle("Machine Tasks")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution:")
            # Assign tasks to machines based on the solution.
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

        # Generate and print the schedule for each machine.
        output = ""
        for machine in all_machines:
            outputTable = assigned_jobs[machine].sort()  # Sort tasks by start time.
            sol_line_tasks = "Machine " + str(machine) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                # Format the task information for printing.
                sol_line_tasks += f"{name:15}"
                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = f"[{start},{start + duration}]"
                sol_line += f"{sol_tmp:15}"
                #Add a bar to the graph
                axis.barh(machine ,width=duration-start ,left=start, color=colours[assigned_task.job], label="Task:" + str(assigned_task.index))

            sol_line += "\n"
            sol_line_tasks += "\n"
            output += sol_line_tasks
            output += sol_line

        print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
        print(output)
    else:
        print("No solution found.")

    # Print statistics about the solution process.
    print("\nStatistics")
    print(f"  - conflicts: {solver.NumConflicts()}")
    print(f"  - branches : {solver.NumBranches()}")
    print(f"  - wall time: {solver.WallTime()}s")

    #Add y axis label and show graph
    axis.set_yticks(range(len(jobs_data)))
    axis.set_yticklabels([f'Machine:{i+1}' for i in range(len(jobs_data))])
    plt.show()

#Starts the program
if __name__ == "__main__":
    main()