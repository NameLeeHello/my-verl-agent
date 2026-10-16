
# conda activate verl-agent
# cd autodl-tmp/verl-agent
# HF_ENDPOINT=https://hf-mirror.com bash examples/grpo_trainer/run_alfworld.sh > train.log 2>&1

# tensorboard --logdir=/root/tf-logs/tensorboard_log --port=6007 --bind_all
# tail -f train.log

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS


# 强制中断
# ray stop --force
# pkill -f ray
# pkill -f python
# pkill -f vllm

# 如果有多张显卡，可以调整
# trainer.n_gpus_per_node=1
# actor_rollout_ref.rollout.tensor_model_parallel_size=1


# 其他设置修改：
    # actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \  # 原32
    # actor_rollout_ref.actor.ppo_mini_batch_size=128 \ # 原256
    # trainer.logger=['console','wandb'] \  # 现改为tensorboard
    # actor_rollout_ref.actor.optim.lr=1e-5 \   原1e-6
    # actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \    # 原0.6
    # actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \


# 新增
    # actor_rollout_ref.model.lora_rank=64 \
    # actor_rollout_ref.model.lora_alpha=128 \

    
# 建立tensorboard软链接
ln -s /root/autodl-tmp/verl-agent/tensorboard_log /root/tf-logs
# 设置ray解除物理资源限制
# ray start --head --num-gpus=8 --num-cpus=14 --temp-dir="/root/autodl-tmp/.cache/ray"

# 下载路径迁移到数据盘
export HF_HOME="/root/autodl-tmp/.cache/huggingface"
export RAY_TMPDIR="/root/autodl-tmp/.cache/ray"

# 原0.1
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=8
val_data_size=32
group_size=8

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=[console,tensorboard] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_qwen2.5_1.5b_lora' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True $@
