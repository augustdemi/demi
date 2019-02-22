kshot=1
alpha=0.005
num_updates=5
beta=0.005
init_weight=False
train_start_idx=0
meta_batch_size=14
dir=logs/result/out/${kshot}shot.uplr${alpha}.num_up${num_updates}.metalr${beta}.init_weight${init_weight}.meta_batch_size${meta_batch_size}

if [[ ! -e $dir ]]; then
    mkdir $dir
fi

nohup python -u main.py \
	--metatrain_iterations=4000 \
	--update_batch_size=${kshot} \
	--update_lr=${alpha} \
	--num_updates=${num_updates} \
	--meta_lr=${beta} \
	--init_weight=${init_weight} \
	--datadir=/home/ml1323/project/robert_data/DISFA/kshot_rest/kshot \
        --train_start_idx=${train_start_idx} \
	--meta_batch_size=${meta_batch_size} \
	--num_classes=2 \
	--datasource=disfa \
	--logdir=logs/disfa/ \
	>> ${dir}/train.txt & 

