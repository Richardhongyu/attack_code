#!/usr/bin/env python3
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('..')
# from DFR import DFR_model
import argparse
import numpy as np
import pandas as pd
import pickle


from happy import happy_model
from DFR import DFR_model
#from lenet import LeNet

# Custom Networks
# from networks.lenet import LeNet
# from networks.pure_cnn import PureCnn
# from networks.network_in_network import NetworkInNetwork
# from networks.resnet import ResNet
# from networks.densenet import DenseNet
# from networks.wide_resnet import WideResNet
# from networks.capsnet import CapsNet
# from networks.happy import happy_model
# Helper functions
from differential_evolution import differential_evolution
import helper 

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class PixelAttacker:
    def __init__(self, models, data, class_names, dimensions=(28, 28)):
        # Load data and model
        self.models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models(self.models, self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        target_class = int(target_class)
        predictions = model.predict(imgs_perturbed)[:, target_class]
        #print(predictions)

        # print('================================================================================')
        # print('target_class is')
        # print(target_class)
        # # This function should always be minimized, so return its complement if needed
        # print('predictions shape is')
        # print(predictions.shape)
        # print('================================================================================')
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or
        # targeted classification), return True
        if (verbose):
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img, model, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False, plot=False):
        # Change the target class based on whether this is a targeted attack or not
        
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img, 0]
        # print("----------------------------------------------------",self.y_test[img,0])
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++",target_class)
        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        # bounds = [(0,dim_x), (0,dim_y), (0,256), (0,256), (0,256)] * pixel_count
        bounds = [(0, dim_x), (0, dim_y), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        predict_fn = lambda xs: self.predict_classes(
            xs, self.x_test[img], target_class, model, target is None)
        # print(predict_fn)
        callback_fn = lambda x, convergence: self.attack_success(
            x, self.x_test[img], target_class, model, targeted_attack, verbose)
        # print(callback_fn)
        #print('模型的输入:',predict_fn,'类型为:',predict_fn.type)
        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[img])[0]
        #print('***********************************************')
        #print('attack_image',attack_image)
        temp_list = attack_image.tolist()
        
        #print('***********************************************')
        # Calculate the L2 norm to represent the revise size
        L2_img = attack_image - self.x_test[img]
        L2_img = np.array(L2_img)

        L2_img = L2_img.reshape(784)
        # print(L2_img)
        L2_norm = np.sqrt(np.sum(np.square(L2_img)))

        prior_probs = model.predict(np.array([self.x_test[img]]))[0]
        # print('-----------------------_test1', prior_probs)
        predicted_probs = model.predict(np.array([attack_image]))[0]
        # print('-----------------------_test2', predicted_probs)
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        return [model.name, pixel_count, img, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x, L2_norm,attack_image]
        # return attack_image

    def attack_all(self, models, samples=10000, pixels=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), targeted=False,
                   maxiter=75, popsize=400, verbose=False):
        attack_images = []
        results = []
        for model in models:
            model_results = []
            valid_imgs = self.correct_imgs[self.correct_imgs.name == model.name].img

#             img_samples = np.random.choice(valid_imgs, samples)
            img_samples = valid_imgs
            for pixel_count in pixels:
                for i, img in enumerate(img_samples):
                    print(model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                    targets = [None] if not targeted else range(10)

                    for target in targets:
                        if (targeted):
                            print('Attacking with target', class_names[target])
                            if (target == self.y_test[img, 0]):
                                continue
                        result = self.attack(img, model, target, pixel_count,
                                             maxiter=maxiter, popsize=popsize,
                                             verbose=verbose)
                        print('------------------------------------')
                        if(result[5]):
                            attack_images.append(result[11])

                        print('------------------------------------')
                        model_results.append(result)

            results += model_results
            #helper.checkpoint(results, targeted)
        return results,attack_images


if __name__ == '__main__':
    # 如果你想要添加你自己的模型
    model_defs = {
      #  'lenet': LeNet,
        # 'pure_cnn': PureCnn,
        # 'net_in_net': NetworkInNetwork,
        # 'resnet': ResNet,
        # 'densenet': DenseNet,
        # 'wide_resnet': WideResNet,
        # 'capsnet': CapsNet,
        'DFR':DFR_model,
        'happy': happy_model,
    }

    parser = argparse.ArgumentParser(description='Attack models on Cifar10')
    parser.add_argument('--model', nargs='+', choices=model_defs.keys(), default=model_defs.keys(),
                        help='Specify one or more models by name to evaluate.')
    parser.add_argument('--pixels', nargs='+', default=[1,3,5,10,15,20,25,30], type=int,
                        help='The number of pixels that can be perturbed.')
    parser.add_argument('--maxiter', default=75, type=int,
                        help='The maximum number of iterations in the differential evolution algorithm before giving up and failing the attack.')
    parser.add_argument('--popsize', default=400, type=int,
                        help='The number of adversarial images generated each iteration in the differential evolution algorithm. Increasing this number requires more computation.')
    parser.add_argument('--samples', default=3, type=int,
                        help='The number of image samples to attack. Images are sampled randomly from the dataset.')
    parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
    parser.add_argument('--save', default='results.pkl', help='Save location for the results (pickle)')
    parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

    args = parser.parse_args()

    # Load data and model
    # Load data and model
    # _, test = cifar10.load_data()

    # data_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/data_norm8CLS_ALL.npy'
    # label_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/label_norm8CLS_ALL.npy'
    # data = np.load(data_path)
    # label_n = np.load(label_path)
    # print(label_n[1:10])
    # print('label的数量', label_n.shape)
    # print("label的格式为：", type(label_n))
    # data = data.reshape([-1, 28, 28, 1])
    # x_train = data[:12417]  # 2484
    # y_train = label_n[:12417]
    # x_test = data[12417:]
    # y_test = label_n[12417:]

    data_path = '../0_AEEA_dataset/MarewareAttack_for_traffic_dataset/data_Mrest.npy'
    label_path = '../0_AEEA_dataset/MarewareAttack_for_traffic_dataset/label_Mrest.npy'
    data = np.load(data_path)
    label_n = np.load(label_path)
    print(label_n[1:10])
    print('data的数量', data.shape)
    print('label的数量', label_n.shape)
    print("label的格式为：", type(label_n))
    data = data.reshape([-1, 28, 28, 1])
    x_test = data[:10000]
    y_test = label_n[:10000]

    # print(x_test[0])
    # for i in x_test:
    #     big_pixel = i[:,:,0]
    #     # print(big_pixel)
    #     i = 256*big_pixel
    #     i = i.astype(int)
    x_test = x_test * 256
    x_test = x_test.astype(int)
    #x_train = x_train * 256
   # x_train = x_train.astype(int)
    y_test = y_test.astype(int)
    #y_train = y_train.astype(int)

    # print(x_test[0])
    # for i in y_test:
    #     big_pixel = i[:,:0]
    #     i = int(256*big_pixel)
    # for i in x_train:
    #     big_pixel = i[:,:0]
    #     i = int(256*big_pixel)
    # for i in y_train:
    #     big_pixel = i[:,:0]
    #     i = int(256*big_pixel)
    test = (x_test, y_test)
    print(len(test))
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    models = [model_defs[m](load_weights=True) for m in args.model]

    attacker = PixelAttacker(models, test, class_names)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print((attacker.correct_imgs))
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    print('Starting attack')

    results,attack_images = attacker.attack_all(models, samples=args.samples, pixels=args.pixels, targeted=args.targeted,
                                  maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
    # attack_images = []
    # if(results[5]):
    #     attack_images.append(results[11])
    attack_images = np.array(attack_images)
    np.save("AE_DFR_10000_3point.npy",attack_images)

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs',
              'perturbation', 'L2_Norm','attack_image']
    results_table = pd.DataFrame(results, columns=columns)

    print(results_table[['model', 'pixels', 'image', 'true', 'predicted', 'success', 'L2_Norm']])

#     print('Saving to', args.save)
#     with open(args.save, 'wb') as file:
#         pickle.dump(results, file)

    # model=load_model('networks/models/lenet.h5')
    #
    # print(model.layers)
    # print(model)
