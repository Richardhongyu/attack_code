# Attack code

## What can I do with these files?

1. You can use attack_for_AE/log/traffic.py to attack different datasets with different models in DE.
2. You can find different models in the DFR/happy/lenet/DFR_log/happy_log/lenet_log.py.
3. Meanwhile you can randomly attactk models by random_arrack/random_generate/random_generate_for_log.py.

## where can I get the pretrained models and datasets?

1. [url](https://pan.baidu.com/s/1z6F8n5GpKqA2yjRtY-Uojw)
    password:6mw8

2. The composition of the directoty in the url:
    - 0_AEEA_dataset
    Our datasets are here including a log dataset,a traffic dataset.
    - 1_model_for_traffic
    This directory includes our pretrained models in traffic dataset. You can get the training process by the tensorboard.
    - 2_model_for_log
    This directory includes our pretrained models in traffic dataset. You can get the training process by the tensorboard.
    - 3_attack_code
    You can get all the training code in this directory.
    If you want to test your own model, you can add your model in the your_model_name.py and put your pretrained model here.
    You can also try different ways to attck models, such as random attack,differential evolution.
    It is convinient to try your models in different dataset.
    - 4_EVALUATION
    - 5_For_TEST_h5
    You can get the accurate attack resutls.

## How can I attack my models?

- You need to pass your args to attack_for_traffic.py(or other attack files) to attack models. 
>EN:You need to train your own models before you attack it.
- Example: python model_name.py --model model_name --other_args
- To get more args, you can read attack_for_traffic.py.

## Envirionment

tensorflow_1_13_gpu 
keras

## Some important tips

- You can write your own model that you want to attack in keras. And you need to follow the examples in the happy.py.
- I write some comments for attacking models in the attack_for_traffic.py.
- I write some comments for model building in the happy.py.
- You can find the specific implements of differential_evolution in the differential_evolution.py.
