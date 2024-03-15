import random
import pandas as pd

num_jobs = 20
num_machines = 9
current_task = 1

# Generate random number of tasks for each job (between 1 and 5), machine to complete on, time (between 2 and 5 hours)
tasks_per_job = [random.randint(1, 5) for i in range(num_jobs)]
total_jobs = sum(tasks_per_job)
machine_numbers = [random.randint(0, num_machines) for i in range(total_jobs)]
task_times = [random.randint(2, 5) for i in range(total_jobs)]

# Create dataframe to store the dataset
data = {'Job': [], 'Task': [], 'Machine': [], 'Time': []}

# Populate dataframe with generated data
for job, num_tasks in enumerate(tasks_per_job, start=1):
    for task in range(1, num_tasks + 1):
        data['Job'].append(job)
        data['Task'].append(task)
        data['Machine'].append(machine_numbers[current_task - 1])
        data['Time'].append(task_times[current_task - 1])
        current_task += 1

# Convert to pandas dataframe
df = pd.DataFrame(data)

# Save dataframe to CSV file without index
df.to_csv('rota_scheduling_dataset.csv', index=False)