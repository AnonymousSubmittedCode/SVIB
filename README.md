# SVIB
Implementation of the Information Bottleneck in reinforcement learning with stein variational gradient<br>
The implementation is based on the openai-baselines.<br>
To run A2C with our framework(SVIB with uniform distribution, just run<br>
`python3 train_use_a2c_svib.py --env='PongNoFrameskip-v4' --train_option='uniform' --num_timesteps=7000000 --ib_alpha=0.001 --lrschedule='constant'`<br>
For other games like Qbert, run<br>
`python3 train_use_a2c_svib.py --env 'QbertNoFrameskip-v4' --num_timesteps=14000000 --lrshedule='double_linear_con' --train_option='uniform' --num_timesteps=7000000 --ib_alpha=0.001`<br>
The parameter 'ib_alpha' is \beta in the paper.<br>
For ppo, run<br>
`python3 train_use_ppo_svib.py --env='QbertNoFrameskip-v4' --train_option='uniform' --num_timesteps=10000000 --ib_alpha=0.0016`<br>
If you want to train agents in Pong, run<br>
`python3 train_use_ppo_svib.py --env='PongNoFrameskip-v4' --train_option='uniform' --num_timesteps=5000000 --ib_alpha=0.0016`<br>
We also implement variational information bottleneck in reinforcement learning, run<br>
`python3 train_use_a2c.py --env='SeaquestNoFrameskip-v4' --train_option='VIB' --beta=0.001`<br>
Also, all required python packages are listed in the requirements.txt, just run<br>
`pip3 install -r requirements.txt` to install these packages.
