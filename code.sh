data_name=pick_place_conveyor

CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
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
--max_steps 100005 \
--save_freq 10000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 1 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--run_id_note VLA-Adapter--pick_place_conveyor--$current_time



python policy_server.py \
  --pretrained_checkpoint outputs/configs+pick_place_conveyor+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--pick_place_conveyor----90000_chkpt \
  --host 0.0.0.0 \
  --port 8000 \
  --debug