##################################################
## Author: RuochenLiu
## Email: ruochen.liu@columbia.edu
## Version: 1.0.0
##################################################
import time
import numpy as np
import cv2

def get_one_hot(targets, n_classes):
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[n_classes])

def stand(img):
    return (img - np.mean(img))/np.std(img)

def preprocess_data():
    img_size = 128
    num_each = 12500
    imgs = []
    for i in range(num_each):
        img_cat = cv2.resize(cv2.imread('../data/train/cat.{}.jpg'.format(i)), (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        img_dog = cv2.resize(cv2.imread('../data/train/dog.{}.jpg'.format(i)), (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        img_cat = np.reshape(cv2.cvtColor(img_cat, cv2.COLOR_BGR2GRAY), (img_size, img_size, 1))
        img_dog = np.reshape(cv2.cvtColor(img_dog, cv2.COLOR_BGR2GRAY), (img_size, img_size, 1))
        imgs.append(img_cat/255)
        imgs.append(img_dog/255)
        print("{} == Processing images: {}%".format(time.asctime(time.localtime(time.time())), np.round((i+1)*100/12500, 2)), end="\r")
    X = np.stack(imgs, axis=0)
    y = get_one_hot(np.array([0, 1]*num_each), 2)
    print("\n")
    np.save("../data/X.npy", X)
    np.save("../data/y.npy", y)

def main():
    preprocess_data()

if __name__ == "__main__":
    main()