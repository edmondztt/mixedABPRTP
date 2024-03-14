#!/bin/bash

tail -n +2 parameters.csv > tmp

python_script="kinesis.py"

while IFS=, read -r Np runtime Q0 Q1 kT2 kH2 kS2
do
    echo "Running ${python_script} with parameters: $Np $runtime $Q0 $Q1 $kT2 $kH2 $kS2"

    source /home/wanxuan/venvpheromone/bin/activate
    # Run the Python script with parameters
    logname="log${Np}-${runtime}-${Q0}-${Q1}-${kT2}-${kH2}-${kS2}"
    echo "logname is ${logname}"
    nohup python -u "$python_script" "$Np" "$runtime" "$Q0" "$Q1" "$kT2" "$kH2" "$kS2" > ${logname} &

    # Wait for the Python script to finish
    # wait

    # echo "${python_script} finished."
done < tmp