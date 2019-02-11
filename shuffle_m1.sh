gpu='1'
meta_batch_size=14
kshot_seed=0
vae_name='allaus_iter200'
vae_model='vae_log/deep/h5/'${vae_name}
num_au=8
alpha=0.01
num_updates=1
beta=0.01
sbjt_start_idx=0
feature_path='/home/ml1323/project/robert_data/DISFA/deep_feature200/'
lambda2=0.1
shf_bat=1
opti=adadelta
model=m1_ce_${lambda2}co_shuffle${shf_bat}_${opti}

echo "$gpu"

kshot=10

echo "${model}"

#dir=./new/disfa/seed${kshot_seed}/m1_ce_${lambda2}reg2/
#if [[ ! -e $dir ]]; then
#  mkdir $dir
#fi

dir=./new/seed${kshot_seed}/m1_0203/
if [[ ! -e $dir ]]; then
  mkdir $dir
fi

nohup python -u main_shf.py \
    --opti=${opti} \
    --shuffle_batch=${shf_bat} \
    --model=${model} \
    --lambda2=${lambda2} \
    --resume=False \
    --num_updates=${num_updates} \
    --metatrain_iterations=20000 \
    --update_batch_size=${kshot} \
    --update_lr=${alpha} \
    --sbjt_start_idx=${sbjt_start_idx} \
    --meta_lr=${beta} \
    --datadir=/home/ml1323/project/robert_data/DISFA/deep_feature200/ \
    --labeldir=/home/ml1323/project/robert_data/DISFA/label/ \
    --meta_batch_size=${meta_batch_size} \
    --datasource=disfa \
    --num_classes=2 \
    --logdir=/home/ml1323/project/robert_code/new/disfa/seed${kshot_seed}/${model}/ \
    --vae_model=${vae_model} \
    --kshot_seed=${kshot_seed} \
    --num_au=${num_au} \
    --gpu=${gpu} \
    --au_idx=0 \
    --vae_model=$vae_model \
    --feature_path=${feature_path} \
    > ${dir}/${kshot}shot.uplr${alpha}.num_up${num_updates}.metalr${beta}.meta_batch_size${meta_batch_size}_${model}.txt &
