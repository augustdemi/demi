gpu='4'
init=False
meta_batch_size=18
kshot_seed=0
fold=fold1
vae_name=$fold'_iter100_adadelta_frame'
vae_model='./base_val/'${vae_name}
num_au=8
alpha=0.008
num_updates=1
beta=0.008
sbjt_start_idx=0
lambda2=0.01
shf_bat=1
opti=adam
model=m1_ce_${lambda2}co_shuffle${shf_bat}_${opti}_each
#model=m1_ce_${lambda2}co_${opti}_check_shample
subject_index='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17'
#subject_index='9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26'
#subject_index='0,1,2,3,4,5,6,7,8,18,19,20,21,22,23,24,25,26'
log_folder=validation500
feature_path=/home/ml1323/project/robert_data/DISFA_new/deep_feature_val/${fold}_iter100_adadelta_frame/dim500/train-train/

echo "$gpu"

kshot=10

echo "${model}"

dir=./$log_folder/${fold}/
if [[ ! -e $dir ]]; then
  mkdir $dir
fi
dir=./$log_folder/${fold}/train/
if [[ ! -e $dir ]]; then
  mkdir $dir
fi

nohup python -u main_shf.py \
    --init=$init \
    --feat_dim=500 \
    --subject_index=${subject_index} \
    --check_sample=False \
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
    --datadir=$feature_path \
    --labeldir=/home/ml1323/project/robert_data/DISFA_new/label/ \
    --meta_batch_size=${meta_batch_size} \
    --datasource=disfa \
    --num_classes=2 \
    --logdir=/home/ml1323/project/robert_code/${log_folder}/maml/${fold}/${model}/ \
    --vae_model=${vae_model} \
    --kshot_seed=${kshot_seed} \
    --num_au=${num_au} \
    --gpu=${gpu} \
    --au_idx=0 \
    > ${dir}/${kshot}shot.uplr${alpha}.num_up${num_updates}.metalr${beta}.meta_batch_size${meta_batch_size}_${model}.init${init}.opti${opti}.txt &
