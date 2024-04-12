#!/bin/bash

tail -n +2 parameters.csv > tmp

MAX_JOBS=32
JOB_COUNT=0

python_script="navigation.py"

while IFS=, read -r Np runtime noise_Q kHT2 DR iftaxis ifkk ifok plate_condition iftail depth
do
    echo "Running ${python_script} with parameters: $Np $runtime $noise_Q $kHT2 $DR $iftaxis $ifkk $ifok $plate_condition $iftail $depth"

    source /home/wanxuan/venvpheromone/bin/activate
    # Run the Python script with parameters
    logname="log${Np}-${runtime}-${noise_Q}-${kHT2}-${DR}-${iftaxis}-${ifkk}-${ifok}-${plate_condition}-${iftail}-${depth}"
    echo "logname is ${logname}"
    nohup python -u "$python_script" "$Np" "$runtime" "$noise_Q" "$kHT2" "$DR" "$iftaxis" "$ifkk" "$ifok" "$plate_condition" "$iftail" "$depth" > ${logname} &
    
    ((JOB_COUNT++))
    if [ "$JOB_COUNT" -ge $MAX_JOBS ]; then
        echo "wait for current batch of $JOB_COUNT jobs to finish"
        wait # Wait for all background jobs to finish
        JOB_COUNT=0 # Reset the job count
        echo "now launch next batch of jobs"
    fi

done < tmp

wait

rm tmp
# Initialize n_max to 0
n_max=0

#exit()

# Find the largest n in existing files
for file in parameters.finished.*.csv; do
  if [[ $file =~ parameters.finished.([0-9]+).csv ]]; then
    number=${BASH_REMATCH[1]}
    if (( number > n_max )); then
      n_max=$number
    fi
  fi
done

echo "n max = $n_max"

scp -r data/* tingtao@131.215.127.174:~/myhdd1/pheromone-modeling/data

mv parameters.csv "parameters.finished.$((n_max + 1)).csv"
