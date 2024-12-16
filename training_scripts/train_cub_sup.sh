for seed in 42 40 44
do
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29451' \
	--use_env main.py \
	cub_caprompt \
	--model vit_base_patch16_224 \
	--batch-size 64 \
	--epochs 50 \
	--lr 0.06 \
	--data-path /data/dataset/liqiwei \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--seed $seed \
	--larger_prompt_lr \
	--ca_storage_efficient_method covariance \
	--e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
	--penalty_weight 0.2 \
	--delta_weight 1 \
	--length 5 \
	--output_dir ./output/cub_sup_seed$seed \
	--trained_caprompt_model ./output/load_ckpt_for_eval \
	
done

