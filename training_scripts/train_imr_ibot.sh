for seed in 40 42 44
do
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29501' \
	--use_env main.py \
        imr_caprompt \
        --model vit_base_patch16_224_ibot \
        --batch-size 24 \
        --epochs 100 \
        --data-path /data/dataset/liqiwei \
        --ca_lr 0.005 \
        --crct_epochs 30 \
	--sched constant \
        --seed $seed \
	--length 10 \
        --prompt_momentum 0.0 \
        --prompt_constran linear \
        --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 \
        --penalty_weight 0.2 \
        --larger_prompt_lr \
        --ca_storage_efficient_method covariance \
        --delta_weight 1 \
	--output_dir ./output/imr_ibot_seed$seed  \
        --trained_caprompt_model ./output/load_ckpt_for_eval
done





