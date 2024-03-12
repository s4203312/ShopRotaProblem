"""
Job Shop Scheduling Example.

This Python script demonstrates a basic job shop scheduling problem using Google's OR-Tools.
A job shop involves scheduling jobs on machines. Each job consists of a sequence of tasks,
which must be performed in a given order, each on a specific machine for a specific duration.

This example models a simple scenario where each job has a predefined sequence of tasks,
each task requiring a specific machine and taking a certain amount of time to complete.
The goal is to schedule all tasks such that the total time to complete all jobs (makespan) is minimized,
while ensuring that tasks follow their specific order within each job and no machine processes more than
one task at a time.

The solution involves creating a constraint programming model, defining variables for task start times,
end times, and intervals, adding constraints for machine availability and task sequencing within jobs,
and defining an objective to minimize the makespan. The model is then solved using the CP-SAT solver.

Key Concepts:
- Horizon: The maximum time span considered for scheduling, calculated as the sum of all task durations.
            It provides an upper limit for task scheduling.
- Disjunctive Constraints: Ensure no two tasks are simultaneously processed on the same machine.
- Precedence Constraints: Ensure tasks within a job are completed in the specified order.

"""

import collections  # Provides access to specialized container datatypes.

from ortools.sat.python import cp_model  # Import the CP-SAT solver.


def main():
    
    #Include own dataset form better grade******
    #Mabye try different solvers******
    #Create a visual representation of the output?******
    
    # Data: List of jobs, each job is a list of tasks, and each task is a tuple (machine_id, processing_time).
    jobs_data = [
        [
            (0, 3),
            (1, 2),
            (2, 2),
        ],  # Job0: Tasks (Machine 0 for 3 units, Machine 1 for 2 units, Machine 2 for 2 units)
        [
            (0, 2),
            (2, 1),
            (1, 4),
        ],  # Job1: Tasks (Machine 0 for 2 units, Machine 2 for 1 unit, Machine 1 for 4 units)
        [(1, 4), (2, 3)],  # Job2: Tasks (Machine 1 for 4 units, Machine 2 for 3 units)
    ]

    print(jobs_data)

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

    # Solve the model using the CP-SAT solver.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

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
            assigned_jobs[machine].sort()  # Sort tasks by start time.
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


if __name__ == "__main__":
    main()