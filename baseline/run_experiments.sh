#!/bin/bash

# Array of parameters
params=("SiO2" "TiO2" "Al2O3" "FeOT" "MgO" "CaO" "Na2O" "K2O")
models=("lasso" "ridge" "elasticnet")

# Function to run a command in the background and log its start time
run_experiment() {
    local param=$1
    local models_str=""
    for model in "${models[@]}"; do
        models_str+="-m $model "
    done
    echo "$(date): Starting experiment with -o $param and models ${models[*]}" >> master_log_ngb.txt
    nohup venv/bin/python3.12 experiments/optuna_run.py -n 200 -o $param $models_str > optuna_log_ngb_$param.txt 2>&1 &
    echo $! >> pids.txt  # Record the PID of the background process
}

# Function to check the exit status of a process
check_status() {
    local pid=$1
    local param=$2
    wait $pid
    local status=$?
    if [ $status -ne 0 ]; then
        echo "$(date): Experiment with -o $param failed with status $status" >> master_log_ngb.txt
    else
        echo "$(date): Experiment with -o $param completed successfully" >> master_log_ngb.txt
    fi
}

# Run the first two commands
run_experiment ${params[0]}
run_experiment ${params[1]}

# Remove the first two elements from the array
params=("${params[@]:2}")

# Loop to wait for a job to finish and then start the next one
while [ ${#params[@]} -gt 0 ]; do
    # Wait for any background process to finish
    pid=$(head -n 1 pids.txt)
    param=${params[0]}
    check_status $pid $param

    # Remove the processed PID and parameter from the lists
    sed -i '1d' pids.txt
    params=("${params[@]:1}")

    # Run the next command if any remain
    if [ ${#params[@]} -gt 0 ]; then
        run_experiment ${params[0]}
    fi
done

# Wait for all remaining background jobs to finish and check their statuses
while read -r pid; do
    param=${params[0]}
    check_status $pid $param
    params=("${params[@]:1}")
done < pids.txt

# Clean up the PID file
rm pids.txt
