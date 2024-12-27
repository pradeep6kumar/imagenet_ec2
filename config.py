config = {
    'checkpoint_dir': '/home/ubuntu/imagenet_ec2/checkpoints',  # Update with your EC2 path
    'batch_size': 256,
    'learning_rate': 0.001 * 2,
    'epochs': 30,
    'num_workers': 8,
    'checkpoint_frequency': 5,
    # Optimizer parameters
    'weight_decay': 1e-4,
    # Additional parameters
    'warmup_epochs': 5,
    'grad_clip_value': 1.0,
    'early_stopping_patience': 10,
    'early_stopping_delta': 0.001,
    # OneCycleLR specific parameters
    'cycle_momentum': True,
    'base_momentum': 0.85,
    'max_momentum': 0.95,
    'div_factor': 25,
    'final_div_factor': 1e4,
    'prefetch_factor': 4,
    'pin_memory': True,
} 