kshot=5
alpha=0.005
num_updates=5
beta=0.005
init_weight=True
max=26
num_test_pts=1
test_iter=0
dir=logs/result/out/${kshot}shot.uplr${alpha}.num_up${num_updates}.metalr${beta}.init_weight${init_weight} 

if [[ ! -e $dir ]]; then
    mkdir $dir
fi

for i in `seq 14 $max`
do
	echo "$i"
	nohup python -u main.py \
		--update_batch_size=${kshot} \
		--test_iter=${test_iter} \
		--num_test_pts=${num_test_pts} \
		--meta_lr=${beta} \
		--update_lr=${alpha} \
		--num_updates=${num_updates} \
		--train=False \
		--test_set=True \
		--subject_idx=$i \
		--init_weight=${init_weight} \
		--datadir=/home/ml1323/project/robert_data/DISFA/kshot_rest/kshot \
		--num_classes=2 \
		--datasource=disfa \
		--meta_batch_size=14 \
		--logdir=logs/disfa/ \
		>> ${dir}/test.metaIter${test_iter}.txt &
	WORK_PID=`jobs -l | awk '{print $2}'`
	wait $WORK_PID
done



