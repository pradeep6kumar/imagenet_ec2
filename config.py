config = {
    'checkpoint_dir': '/home/ubuntu/imagenet_ec2/checkpoints',
    'batch_size': 256,
    'learning_rate': 0.001,  # Reduced base learning rate
    'epochs': 100,  # Increased from 30 to 100
    'num_workers': 8,
    'checkpoint_frequency': 5,
    'weight_decay': 1e-4,
    'warmup_epochs': 10,  # Increased warmup period
    'grad_clip_value': 1.0,
    'early_stopping_patience': 15,
    'early_stopping_delta': 0.001,
    'cycle_momentum': True,
    'base_momentum': 0.85,
    'max_momentum': 0.95,
    'div_factor': 10,
    'final_div_factor': 1e4,
    'prefetch_factor': 4,
    'pin_memory': True,
    'gradient_accumulation_steps': 4
} 