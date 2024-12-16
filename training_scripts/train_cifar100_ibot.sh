for seed in 42 40 44
do
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29401' \
	--use_env main.py \
	cifar100_caprompt \
	--model vit_base_patch16_224_ibot \
	--batch-size 24 \
	--epochs 50 \
	--data-path /data/dataset/liqiwei \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--seed $seed \
	--length 5 \
	--sched step \
	--e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
	--larger_prompt_lr \
	--ca_storage_efficient_method covariance \
	--penalty_weight 0.0 \
	--delta_weight 0.1 \
	--output_dir ./output/cifar100_ibot_seed$seed  \
	--trained_caprompt_model ./output/load_ckpt_for_eval \
	
done