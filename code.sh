data_name=merged_rmbench
dinov3_feature_dir=/mnt/lx/cyx/lerobot/dinov3_features/$data_name
dinov3_rlds_spec=/mnt/lx/cyx/lerobot/dinov3_features/${data_name}.rlds_spec.pkl

# Step 1 (VLA / Prismatic env): export RLDS kwargs + dataset_statistics into a pickle.
CUDA_VISIBLE_DEVICES=2 python vla-scripts/export_dinov3_rlds_spec.py \
--data_root_dir /mnt/lx/cyx/lerobot/dataset \
--dataset_name 'remanipbench_rlds' \
--output /mnt/lx/cyx/lerobot/dinov3_features/remanipbench_rlds.rlds_spec.pkl

# Step 2 (DINOv3-friendly env, no Prismatic): precompute features from the spec.
CUDA_VISIBLE_DEVICES=2 python vla-scripts/precompute_dinov3_features.py \
--spec_pickle /mnt/lx/cyx/lerobot/dinov3_features/remanipbench_rlds.rlds_spec.pkl \
--output_dir /mnt/lx/cyx/lerobot/dinov3_features/remanipbench_rlds \
--resize_resolution 224,224 \
--model_id ./dinov3-vitl16-pretrain-lvd1689m \
--batch_size 64

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node 6 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir /mnt/lx/cyx/lerobot/dataset \
--dataset_name 'remanipbench_rlds' \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--num_temporal_frames 2 \
--temporal_frame_interval 1 \
--temporal_fusion_type attention \
--use_current_query_temporal_attention False \
--use_mid_layer_temporal_fusion True \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 100000 \
--max_steps 80005 \
--save_freq 10000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 1 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--use_future_pred True \
--use_future_conf False \
--future_confidence_gamma 1.0 \
--future_conf_loss_weight 0.1 \
--pred_tokens_before_action False \
--future_pred_feature_dir /mnt/lx/cyx/lerobot/dinov3_features \
--future_pred_loss_weight 0.05 \
--use_latency_conditioning False \
--latency_steps_min 0 \
--latency_steps_max 6 \
--run_id_note VLA-Adapter--remanipbench_rlds--pred--2frame--attn--mid_layer--$current_time \
--use_relative_action false \
--relative_action_mask true,true,true,true,true,true,true,false



python policy_server.py \
  --pretrained_checkpoint outputs/configs+pick_place_conveyor+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--pick_place_conveyor--bs8----80000_chkpt \
  --host 0.0.0.0 \
  --port 8000


python policy_server.py \
  --pretrained_checkpoint outputs/configs+pick_place_conveyor+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--relative_action--VLA-Adapter--pick_place_conveyor----100000_chkpt \
  --use_relative_action \
  --relative_action_mask true,true,true,true,true,true,true,false \
  --host 0.0.0.0 \
  --port 8000


python policy_server.py \
  --pretrained_checkpoint outputs/configs+pick_place_conveyor+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--pick_place_conveyor--pred2----60000_chkpt \
  --use_future_pred \
  --host 0.0.0.0 \
  --port 8000



python policy_server.py \
  --pretrained_checkpoint outputs/configs+rmbench+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--rmbench--pred--2frame--attn--mid_layer----60000_chkpt \
  --use_future_pred \
  --num_temporal_frames 2 \
  --temporal_fusion_type attention \
  --use_mid_layer_temporal_fusion \
  --unnorm_key ballcatching_rlds \
  --use_cuda_graph \
  --host 0.0.0.0 \
  --port 8000




python policy_server.py \
  --pretrained_checkpoint outputs/configs+pick_place_conveyor+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--pick_place_conveyor--pred--2frame--attn--mid_layer--latency----60000_chkpt \
  --use_future_pred \
  --num_temporal_frames 2 \
  --temporal_fusion_type attention \
  --use_mid_layer_temporal_fusion \
  --use_latency_conditioning \
  --latency_steps 2 \
  --latency_steps_max 6 \
  --host 0.0.0.0 \
  --port 8000



torchrun --standalone --nproc_per_node=4 vla-scripts/train_recon_probe.py \
  --pretrained_checkpoint outputs/configs+pick_place_conveyor+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--pick_place_conveyor--pred2----60000_chkpt \
  --data_root_dir /mnt/lx/cyx/lerobot/dataset \
  --batch_size 8 \
  --max_steps 80005 \
  --save_freq 10000 \
  --dataset_name pick_place_conveyor




python vla-scripts/visualize_temporal_stride_samples.py \
  --data_root_dir /mnt/lx/cyx/lerobot/dataset \
  --dataset_name pick_place_conveyor \
  --output_dir temporal_stride_vis \
  --num_temporal_frames 4 \
  --intervals 1,2,3,4,5,6,7,8 \
  --num_examples 2


CUDA_VISIBLE_DEVICES=0 python vla-scripts/benchmark_inference_latency.py \
  --pretrained_checkpoint outputs/configs+pick_place_conveyor+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--pick_place_conveyor--pred--2frame--attn--mid_layer----60000_chkpt \
  --device cuda:0 \
  --use_future_pred \
  --num_temporal_frames 2 \
  --use_mid_layer_temporal_fusion \
  --use_cuda_graph \
  --warmup 10 \
  --iters 100 \
  --batch_size 1 \
  --output_json logs/latency_benchmark.json-++