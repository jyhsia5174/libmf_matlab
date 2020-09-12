echo "check logs permission"

solver='gauss'
env='gpu'
data_name='ml1m'

d=40
tr='tr'
va='te'

max_iter=(3 3 3)
eps=1e-5

lambda=(0.5 0.05 0.005)

log_path="logs/${solver}_${env}/${data_name}"
mkdir -p $log_path

#task(){
for i in ${!lambda[*]}; do
    lambda_U=${lambda[$i]}
    lambda_V=${lambda[$i]}
    log_name="${data_name}_${d}_${lambda_U}_${max_iter[$i]}.txt"
    matlab -nodisplay -nosplash -nodesktop -r "solver='${solver}';env='${env}';epsilon=${eps};lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';max_iter=${max_iter[$i]};run('example.m');exit;"  > $log_path/$log_name 
done
#
#task
#wait

#task | xargs -0 -d '\n' -P 5 -I {} matlab {} &
