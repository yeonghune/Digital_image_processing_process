import numpy as np
from sklearn.utils import shuffle
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import StratifiedKFold




face_images=[]
face_labels=[]

Index_name=[]

figure_data = 'data'
for idex, figure_datas in enumerate(os.listdir(figure_data)):
    
    figure_dir = os.path.join(figure_data, figure_datas)
    Index_name.append(figure_datas)

    for image_name in os.listdir(figure_dir):
        image_path = os.path.join(figure_dir,image_name)
        
        img_gray = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img_gray,cv2.COLOR_BGR2RGB)
        img_gray = cv2.resize(img_gray, dsize=(112,112))/255.0
        #img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
        #print(img_grayscale)
        face_images.append(img_gray)
        face_labels.append(idex)


face_images = np.array(face_images)
face_labels = np.array(face_labels)




target_shape = face_images[0].shape
#print(target_shape)

def append_data(dataset, iterate = 4):
    
    for i in range(0, iterate):
        dataset = np.append(dataset, dataset, axis = 0)

    print('dataset.shape: ', dataset.shape)

    return dataset

def visualize(image_pairs, labels, n = 5, title = "Image Pair Examples"):
    """ Visualize a few pairs """

    def show(ax, image):
        ax.imshow((image * 255).astype(np.uint8))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    print("Hello")
    fig = plt.figure(figsize=(9, 9)) 
    plt.title(title)
    axs = fig.subplots(n, 2)
    for i in range(n):
        show(axs[i, 0], image_pairs[i][0])
        show(axs[i, 1], image_pairs[i][1])
    plt.savefig('plot.jpg')

def make_pairs(x, y):

    #print('y: ', y)


    pairs = []
    labels = []

    # add a matching example
    #for iter in range(0, (num_classes/2)):

    for i in range(0, len(y)):
        for j in range(0, len(y)):
            # add a matching example
            if y[i] == y[j]:
                #print('y[i], y[j]', y[i], y[j])
                pairs += [[x[i], x[j]]]
                labels += [0]
                pairs += [[x[i], x[j]]]
                labels += [0]
                pairs += [[x[i], x[j]]]
                labels += [0]
                pairs += [[x[i], x[j]]]
                labels += [0]
                pairs += [[x[i], x[j]]]
                labels += [0]
                pairs += [[x[i], x[j]]]
                labels += [0]
            # add a non-matching example
            elif y[i] != y[j]:
                # print('y[i], y[j]', y[i], y[j])
                pairs += [[x[i], x[j]]]
                labels += [1]

    #np.save('./pairs.npy', pairs)
    #np.save('./labels.npy', labels)
    return np.array(pairs), np.array(labels).astype("float32")

pairs_train, labels_train = make_pairs(face_images, face_labels)
del face_images
del face_labels


idx = np.arange(pairs_train.shape[0])
np.random.shuffle(idx)




pairs_train = pairs_train[idx]
labels_train = labels_train[idx]

x_train1 = pairs_train[:, 0]
x_train2 = pairs_train[:, 1]

del pairs_train


x_train1 = append_data(x_train1, 3)
x_train2 = append_data(x_train2, 3)

labels_train = append_data(labels_train, 3)

print(Index_name)