#!/bin/bash
dset=disfa

rm -rf model_input/${dset}
rm -rf model_output/${dset}
rm -rf model_vae/${dset}
rm -rf var_path model_gp/${dset}

mkdir model_input/${dset}
mkdir model_output/${dset}
mkdir model_vae/${dset}
mkdir var_path model_gp/${dset}


current=1
next=$((current+1))

python train_deep_coder.py \
        -i init \
        -n 5 \
        -w 0 \
        -tr DATA_1/${dset}_te.h5 \
        -te DATA_1/${dset}_te.h5 \
        -o model_output/${dset}/${current}


python vgpae_warming/test_deep_autoencoder.py \
        --kl_thres 15 \
        --train model_output/${dset}/${current}_va.h5 \
        --test model_output/${dset}/${current}_te.h5 \
        --n_train 5000 \
        --n_test 5000 \
        --n_latent 50 \
        --kl_reg ${current} \
        --features feat \
        --output_reconstr model_input/${dset}/${next} \
        --output_evaluation model_output/${dset}/eval_${current}.txt \
        -t 5 \
        --save_outfile model_gp/${dset}/gp_${next} 

for current  in `seq 2 30`;
do
        next=$((current+1))
        echo $current $next

        python train_deep_coder.py \
                -n 5 \
                -i model_input/${dset}/${current} \
                -o model_output/${dset}/${next} \
                -w ${current} \
                -tr DATA_1/${dset}_te.h5 \
                -te DATA_1/${dset}_te.h5 \

        python vgpae_warming/test_deep_autoencoder.py \
                --train model_output/${dset}/${current}_va.h5 \
                --test model_output/${dset}/${current}_te.h5 \
                --kl_thres 15 \
                --n_train 5000 \
                --n_test 5000 \
                --kl_reg ${current} \
                --n_latent 50 \
                --var_path model_gp/${dset}/gp_${current} \
                --features feat \
                --output_reconstr model_input/${dset}/${next} \
                --output_evaluation model_output/${dset}/eval_${current}.txt \
                -t 5 \
                --save_outfile model_gp/${dset}/gp_${next} 
done
