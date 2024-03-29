#!/bin/bash

tail -n +2 parameters.csv > tmp

python_script="navigation.py"

while IFS=, read -r Np runtime gamma0_inv Q0 Q1 kT2 kH2 kS2 iftaxis ifkk ifok
do
    echo "Running ${python_script} with parameters: $Np $runtime $gamma0_inv $Q0 $Q1 $kT2 $kH2 $kS2 $iftaxis $ifkk $ifok"

    source /home/wanxuan/venvpheromone/bin/activate
    # Run the Python script with parameters
    logname="log${Np}-${runtime}-${gamma0_inv}-${Q0}-${Q1}-${kT2}-${kH2}-${kS2}-${iftaxis}-${ifkk}-${ifok}"
    echo "logname is ${logname}"
    nohup python -u "$python_script" "$Np" "$runtime" "$gamma0_inv" "$Q0" "$Q1" "$kT2" "$kH2" "$kS2" "$iftaxis" "$ifkk" "$ifok" > ${logname} &

    # Wait for the Python script to finish
    # wait

    # echo "${python_script} finished."
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

#scp data/* tingtao@131.215.127.174:~/myhdd1/pheromone-modeling/data

#mv parameters.csv "parameters.finished.$((n_max + 1)).csv"
