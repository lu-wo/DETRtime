date=$(date +%y-%m-%d-%H-%M)    
modelname=$'movie_3class'

nohup python main.py \
	--data_path /home/ubuntu/projects/data/ICML_movie_min_segmentation_minseq_500_margin_1_amp_thresh_10000/tensors \
	--backbone inception_time \
	--lr_backbone 1e-4 \
	--nb_filters 16 \
	--use_residuals False \
	--backbone_depth 6 \
	--batch_size 32 \
	--bbox_loss_coef 10 \
	--giou_loss_coef 2 \
	--eos_coef 0.4 \
	--hidden_dim 128 \
	--dim_feedforward 512 \
	--dropout 0.1 \
	--wandb_dir movie \
	--num_queries 30 \
	--lr_drop 50 \
	--output_dir ./runs/"$modelname" &
