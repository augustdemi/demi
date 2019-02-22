#nohup python -u train_deep_coder_disfa.py -g '0' -b 444 -n 200 -sm 'lr0.05' -au 12 -tr /home/ml1323/project/robert_data/DISFA/h5_maml/train.h5 -te /home/ml1323/project/robert_data/DISFA/h5_maml/test.h5 > vae_log/lr0.05.txt &
#nohup python -u train_deep_coder_disfa.py -i 'n' -e 100 -g '3' -dec 0 -log 'w_decoder_batchsize32_allau_lr0.1_iter200' -n 200 \
#nohup python -u train_deep_coder_disfa.py -g '3' -dec 1 -log 'w_decoder_batchsize32_allau_lr0.1_iter200_w1' -n 200 \
aus=('au1' 'au2' 'au4' 'au5' 'au6' 'au9' 'au12' 'au15' 'au17' 'au20' 'au25' 'au26')
au_index=3
gpu='3'
name='one_'${aus[${au_index}]}'_iter300_testtttt'
dir='./vae_log/new/h5/'
#name='wo_decoder_batchsize32_allau_iter300'
nohup python -u train_deep_coder_disfa.py -num_au 1 -g ${gpu} -log $name -n 1 \
 -sm ${dir}${name} -au ${au_index} \
 -tr /home/ml1323/project/robert_data/DISFA/h5_maml/train.h5 -te /home/ml1323/project/robert_data/DISFA/new_dataset/half_test_h5/testtest.h5 > vae_log/new/${name}.txt &


#nohup python -u train_deep_coder_disfa.py -lr 0.05 -rm 'earlystop_lr0.05' -sm 'finetune_test_early_lr0.05' -f 1 -g '2' -b 777 -n 100 -i 'n' -au 12 -tr /home/ml1323/project/robert_data/DISFA/h5_maml/test.h5 -te /home/ml1323/project/robert_data/DISFA/h5_maml/test.h5 > vae_model/finetune_test_early_lr0.05.txt &
##The one I run for the final: nohup python -u train_deep_coder_disfa.py -g '1' -b 77 -n 200 -e 100 -i 'n' -au 12 -tr /home/ml1323/project/robert_data/DISFA/h5_maml/train.h5 -te /home/ml1323/project/robert_data/DISFA/h5_maml/test.h5 > au_all_orig_2.txt &
#nohup python -u train_deep_coder_disfa.py -k 2 -n 200 -au 11 -e 200 -i "n" -tr /home/ml1323/project/robert_data/DISFA/h5_maml/train.h5 -te /home/ml1323/project/robert_data/DISFA/h5_maml/test.h5 > disfa_soft_au26_3.txt &
