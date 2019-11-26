# Attack code

## The composition of the directoty

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

## How to attack your models

- You need to pass your args to attack_for_traffic.py to attack models. EN:You need to try your own models before you attack it.
- Example: python model_name.py --model model_name --others

## Envirionment

tensorflow_1_13_gpu 
keras

## Some important tips

- You can write your own model that you want to attack in keras. And you need to follow the rules in the 3_attack_code/model_name.py. 
- I write some comments for attacking models in the 3_attack_code/attack_for_traffic.py.
- I write some comments for model building in the 3_attack_code/happy.py.
- You can find the specific implements of differential_evolution in the 3_attack_code/differential_evolution.py.
 
## GOOD LUCKY TO YOUR TRAVEL IN AI!