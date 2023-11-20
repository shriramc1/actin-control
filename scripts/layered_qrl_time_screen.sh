#!/bin/bash

create_slurm_script () {


    WRITE_FILE="./actin_${1}_${2}.sbatch"
    if test -f "$WRITE_FILE"; then 
        rm $WRITE_FILE
    fi
    
    printf "#!/bin/bash\n" >> $WRITE_FILE
    printf "#SBATCH --job-name=actin_${1}_${2}\n" >> $WRITE_FILE
    printf "#SBATCH --array=0-50\n" >> $WRITE_FILE
    printf "#SBATCH --time=12:00:00\n" >> $WRITE_FILE
    printf "#SBATCH --partition=hns,normal\n" >> $WRITE_FILE
    printf "#SBATCH --nodes=1\n" >> $WRITE_FILE
    printf "#SBATCH -c 1\n" >> $WRITE_FILE
    printf "#SBATCH --output=%%x_out.out\n" >> $WRITE_FILE
    printf "#SBATCH --error=%%x_err.err\n" >> $WRITE_FILE

    printf "\n\n" >> $WRITE_FILE
    printf "module purge \n" >> $WRITE_FILE


    

    printf "source /home/groups/rotskoff/shriramconda3/etc/profile.d/conda.sh \n" >> $WRITE_FILE
    printf "conda activate cgexp \n" >> $WRITE_FILE

    printf "python layered_qrl_time.py --init_run_num \$SLURM_ARRAY_TASK_ID --target_observables ${1} ${2}\n" >> $WRITE_FILE

}


all_layer_1=(1.0 5.0)
all_layer_2=(1.0 5.0)
for layer_1 in "${all_layer_1[@]}"
do
for layer_2 in "${all_layer_2[@]}"
do
    create_slurm_script $layer_1 $layer_2
    sbatch actin_${layer_1}_${layer_2}.sbatch &
done
done


all_layer_1=(2.0)
all_layer_2=(2.0)
for layer_1 in "${all_layer_1[@]}"
do
for layer_2 in "${all_layer_2[@]}"
do
    create_slurm_script $layer_1 $layer_2
    sbatch actin_${layer_1}_${layer_2}.sbatch &
done
done

all_layer_1=(3.0)
all_layer_2=(3.0)
for layer_1 in "${all_layer_1[@]}"
do
for layer_2 in "${all_layer_2[@]}"
do
    create_slurm_script $layer_1 $layer_2
    sbatch actin_${layer_1}_${layer_2}.sbatch &
done
done

all_layer_1=(4.0)
all_layer_2=(4.0)
for layer_1 in "${all_layer_1[@]}"
do
for layer_2 in "${all_layer_2[@]}"
do
    create_slurm_script $layer_1 $layer_2
    sbatch actin_${layer_1}_${layer_2}.sbatch &
done
done