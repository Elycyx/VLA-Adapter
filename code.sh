data_name=pick_place_conveyor
dinov3_feature_dir=/mnt/lx/cyx/lerobot/dinov3_features/$data_name
dinov3_rlds_spec=/mnt/lx/cyx/lerobot/dinov3_features/${data_name}.rlds_spec.pkl

# Step 1 (VLA / Prismatic env): export RLDS kwargs + dataset_statistics into a pickle.
CUDA_VISIBLE_DEVICES=2 python vla-scripts/export_dinov3_rlds_spec.py \
--data_root_dir /mnt/lx/cyx/lerobot/dataset \
--dataset_name $data_name \
--output $dinov3_rlds_spec

# Step 2 (DINOv3-friendly env, no Prismatic): precompute features from the spec.
CUDA_VISIBLE_DEVICES=2 python vla-scripts/precompute_dinov3_features.py \
--spec_pickle $dinov3_rlds_spec \
--output_dir $dinov3_feature_dir \
--resize_resolution 224,224 \
--model_id ./dinov3-vitl16-pretrain-lvd1689m \
--batch_size 64

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir /mnt/lx/cyx/lerobot/dataset \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
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
--use_future_pred False \
--future_pred_feature_dir $dinov3_feature_dir \
--future_pred_loss_weight 0.05 \
--run_id_note VLA-Adapter--pick_place_conveyor--bs8--$current_time \
--use_relative_action false \
--relative_action_mask true,true,true,true,true,true,true,false



python policy_server.py \
  --pretrained_checkpoint outputs/configs+pick_place_conveyor+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--pick_place_conveyor----90000_chkpt \
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