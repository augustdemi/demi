nohup python -u main.py  --num_classes=2 --datasource=disfa --metatrain_iterations=10 --meta_batch_size=14 --update_batch_size=1 --update_lr=0.4 --num_updates=5 --logdir=logs/disfa/ > train.txt &
