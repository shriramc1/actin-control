#!/bin/bash

create_slurm_script () {


    WRITE_FILE="./actin_${1}_${2}_${3}_${4}_${5}_${6}.sbatch"
    if test -f "$WRITE_FILE"; then 
        rm $WRITE_FILE
    fi
    
    printf "#!/bin/bash\n" >> $WRITE_FILE
    printf "#SBATCH --job-name=actin_${1}_${2}_${3}_${4}_${5}_${6}\n" >> $WRITE_FILE
    printf "#SBATCH --time=48:00:00\n" >> $WRITE_FILE
    printf "#SBATCH --partition=hns,normal\n" >> $WRITE_FILE
    printf "#SBATCH --nodes=1\n" >> $WRITE_FILE
    # printf "#SBATCH -G 1\n" >> $WRITE_FILE
    printf "#SBATCH -c 2\n" >> $WRITE_FILE
    # printf "#SBATCH --gpu_cmode=shared\n" >> $WRITE_FILE
    printf "#SBATCH --output=%%x_out.out\n" >> $WRITE_FILE
    printf "#SBATCH --error=%%x_err.err\n" >> $WRITE_FILE

    printf "\n\n" >> $WRITE_FILE
    printf "module purge \n" >> $WRITE_FILE


    

    printf "source /home/groups/rotskoff/shriramconda3/etc/profile.d/conda.sh \n" >> $WRITE_FILE
    printf "ml cuda/11.3.1 \n" >> $WRITE_FILE
    # printf "nvidia-smi \n" >> $WRITE_FILE
    printf "conda activate cgexp \n" >> $WRITE_FILE

    printf "python train_qrl.py --decision_time ${1} --num_decisions_per_episode ${2} --target_observable ${3} --tau ${4} --num_explore_episodes ${5} --youngs_threshold ${6} --target_observable_name upper_density\n" >> $WRITE_FILE

}


all_tau=(0.005)
all_decision_times=(1)
all_num_decisions_per_episode=(50)
all_target_observable=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)
all_youngs_threshold=(50)
all_num_explore_episodes=(20)
for decision_time in "${all_decision_times[@]}"
do
for num_decisions_per_episode in "${all_num_decisions_per_episode[@]}"
do
for target_observable in "${all_target_observable[@]}"
do
for tau in "${all_tau[@]}"
do
for num_explore_episodes in "${all_num_explore_episodes[@]}"
do
for youngs_threshold in "${all_youngs_threshold[@]}"
do
    create_slurm_script $decision_time $num_decisions_per_episode $target_observable $tau $num_explore_episodes $youngs_threshold
    sbatch actin_${decision_time}_${num_decisions_per_episode}_${target_observable}_${tau}_${num_explore_episodes}_${youngs_threshold}.sbatch &
done
done
done
done
done
done
