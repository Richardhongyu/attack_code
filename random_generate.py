import random
import numpy as np
def Random_Attack(pixel_count):
        random_attack_results=[]
        for i in  range(pixel_count):
            x = random.randint(0,27)
            y = random.randint(0,27)
            grey_value = random.randint(0,255)
            random_attack_results.append(x)
            random_attack_results.append(y)
            random_attack_results.append(grey_value)
        print(random_attack_results)
        random_attack_results = np.array(random_attack_results)
        return random_attack_results

