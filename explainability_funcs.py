import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import History
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
from typing import Tuple, Dict, List, Union


def visualize_samples(data_path: str) -> Tuple[int, Dict[int, str]]:
    """
    Visualizes sample images (one for each class) from a directory and displays class labels.

    Parameters:
    - data_path (str): The path to the directory containing class subdirectories with images.

    Returns:
    - Tuple[int, Dict[int, str]]: A tuple containing the total number of classes and a dictionary
      mapping numerical labels to class names.
    """
    labels = []
    num_classes = len(os.listdir(data_path))

    fig, ax = plt.subplots(4, 6, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i, label_directory in enumerate(os.listdir(data_path)):
        label_path = os.path.join(data_path, label_directory)

        if os.path.isdir(label_path):
            sample_image = os.listdir(label_path)[0]
            image_path = os.path.join(label_path, sample_image)
            img = Image.open(image_path)

            ax[i // 6, i % 6].set_xticks([])
            ax[i // 6, i % 6].set_yticks([])
            ax[i // 6, i % 6].set_title(label_directory)
            ax[i // 6, i % 6].imshow(img)

        labels.append(label_directory)

    fig.delaxes(ax[3, 5])
    fig.suptitle(f'Total number of classes: {num_classes}', fontsize=13)
    plt.show()

    labels_dict = {i: lab for i, lab in enumerate(sorted(labels))}
    return num_classes, labels_dict


def plot_performance(history: History) -> Tuple[float, float]:
    """
    Plot the performance metrics of a neural network based on its training history.

    Parameters:
    - history (tf.keras.callbacks.History): The training history of a neural network model.

    Returns:
    - Tuple[float, float]: A tuple containing the test loss and test accuracy of the model.
    """
    loss, acc = model.evaluate(val, verbose=0)

    metrics = pd.DataFrame({
        "train_accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "val_loss": history.history["val_loss"],
        "train_loss": history.history["loss"],
        "epoch": np.arange(0, 100)
    })

    minimum = metrics[metrics.val_loss == metrics.val_loss.min()].epoch.values[0]
    maximum = metrics.val_loss.max()

    temp = pd.melt(metrics, id_vars=["epoch"],
                   value_vars=['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss'], var_name='Set', value_name='Score')

    temp['Metric'] = temp['Set'].apply(lambda x: x.split('_')[1].capitalize())
    temp['Set'] = temp['Set'].apply(lambda x: x.split('_')[0].capitalize())

    sns.set()
    rel = sns.relplot(temp, x="epoch", y="Score", col="Metric", hue="Set", kind="line", facet_kws={'sharey': False, 'sharex': True})
    rel.fig.suptitle(f'NN FINAL PERFORMANCES: \n Test accuracy: {acc:.2f} \n Test loss: {loss:.2f}', fontsize=14)
    rel.fig.subplots_adjust(top=.8)
    plt.axvline(minimum, color='red', linestyle="--")
    plt.text(minimum+1, maximum-0.1, f'minimum eval loss\nat {minimum} epochs', color="red", fontsize=10)
    plt.show()

    return loss, acc

loss, acc = plot_performance(history)


def sample_bylabel(dataset: tf.data.Dataset) -> Dict[int, tf.Tensor]:
    """
    Extracts one image example for each class from a batched TensorFlow dataset.

    Parameters:
    - dataset (tf.data.Dataset): A batched TensorFlow dataset.

    Returns:
    - Dict[Any, Any]: A dictionary containing one representative image for each class.
    """
    representative_examples = {}

    for image, label in dataset.unbatch():
        label = label.numpy()  # Convert label to a numpy array for easy comparison

        # check if we already have an example for this label
        if label not in representative_examples.keys():
            representative_examples[label] = image

        # Check if we have found one example for each label
        if len(representative_examples) == num_classes:
            break

    return representative_examples


def compute_saliency(input_: tf.Tensor, model: tf.keras.Model) -> np.ndarray:
    """
    Compute the saliency map for a given input image considering the relative model's output.

    Parameters:
    - input_ (tf.Tensor): Input image.
    - model (tf.keras.Model): A TensorFlow neural network model.

    Returns:
    - tf.Tensor: Saliency map.
    """
    input_ = tf.expand_dims(input_, axis=0) # add a dimension to check the expected Tensor dimonsion for the model

    with tf.GradientTape() as gt:
        gt.watch(input_) # track the input image
        logit = model(input_, training=False)
        logit = tf.squeeze(logit)
        class_score = logit[tf.argmax(logit)] # take only the logit for the selected class

    saliency_map = gt.gradient(class_score, input_)
    saliency_map = tf.squeeze(saliency_map)
    saliency_map = np.max(tf.abs(saliency_map), axis=-1)

    return saliency_map


def modified_saliency(input_: tf.Tensor,
                      model: tf.keras.Model) -> np.ndarray:
    """
    Compute the modified saliency map for a given input image considering the relative model's output.
    The modified saliency map is the element-wise moltiplication of an input by its saliency map.

    Parameters:
    - input_ (tf.Tensor): Input image.
    - model (tf.keras.Model): A TensorFlow neural network model.

    Returns:
    - tf.Tensor: Modified saliency map.
    """
    input_ = tf.expand_dims(input_, axis=0)

    with tf.GradientTape() as gt:
        gt.watch(input_)
        logit = model(input_, training=False)
        logit = tf.squeeze(logit)
        class_score = logit[tf.argmax(logit)]

    saliency_map = gt.gradient(class_score, input_)
    saliency_map = tf.squeeze(saliency_map)
    saliency_map = tf.abs(saliency_map)
    squeezed_input = tf.squeeze(input_)
    mod_sm = saliency_map * squeezed_input
    return saliency_map


def normalize_to_image(im: tf.Tensor) -> np.ndarray:
    """
    Normalize the input image tensor to the range [0, 255].

    Parameters:
    - im (tf.Tensor): Input image.

    Returns:
    - np.ndarray: Normalized image.
    """
    im = 255 * (im - tf.reduce_min(im)) / (tf.reduce_max(im) - tf.reduce_min(im))
    return np.array(im, dtype=np.uint8)


def capped_saliency(saliency_map: np.ndarray) -> np.ndarray:
    """
    Cap the saliency map values at the 99-th percentile.

    Parameters:
    - saliency_map (np.ndarray): Saliency map.

    Returns:
    - np.ndarray: Capped saliency map.
    """
    max_val = np.percentile(saliency_map, 99)
    capped_saliency_map = np.array(saliency_map).copy()
    capped_saliency_map[capped_saliency_map > max_val] = max_val
    return capped_saliency_map


def plot_example_sm(ex: np.ndarray,
                    model: tf.keras.Model) -> None:
    """
    Plots the original input image, its saliency map, and the saliency map capped at the 99th percentile.

    Parameters:
    - ex (tf.Tensor): An example tensor representing an image.
    - model (tf.keras.Model): A TensorFlow neural network model.

    Returns:
    - None
    """

    # Compute the saliency map using the 'compute_saliency' function
    saliency_map = compute_saliency(ex, model)

    plt.rcdefaults()
    fig, ax = plt.subplots(1, 3, figsize=(12, 16))

    # Plot the original input image
    ax[0].imshow(np.array(ex, dtype=np.uint8))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Original input")

    # Plot the saliency map
    ax[1].imshow(saliency_map, cmap="gray")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Saliency map")

    # Plot the saliency map capped at 99th percentile
    ax[2].imshow(capped_saliency(saliency_map), cmap="gray")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title("Saliency map capped at 99-th perc.")

    plt.show()


def noisy_saliency(input_: tf.Tensor,
                   model: tf.keras.Model,
                   sigma: int) -> np.ndarray:
    """
    Add noise to the given image, then compute the saliency map for the noisy
    (smoothed) image, considering the relative model's output.

    Parameters:
    - input_ (tf.Tensor): Input image.
    - model (tf.keras.Model): A TensorFlow neural network model.
    - sigma: the noise level to apply (integer between [0,+inf) which represents a percentage)

    Returns:
    - tf.Tensor: Modified saliency map.
    """

    input_ = tf.expand_dims(input_, axis=0)

    with tf.GradientTape() as gt:
        noise = tf.random.normal(input_.shape, mean=0, stddev=sigma)
        noisy_input = input_ + noise
        gt.watch(noisy_input)
        logit = model(noisy_input, training=False)
        logit = tf.squeeze(logit)
        class_score = logit[tf.argmax(logit)]

    saliency_map = gt.gradient(class_score, noisy_input)
    saliency_map = tf.squeeze(saliency_map)
    saliency_map = np.max(tf.abs(saliency_map), axis=-1)
    noisy_input = tf.squeeze(noisy_input)
    return saliency_map


def compute_smooth_saliency(input_: tf.Tensor,
                            model: tf.keras.Model,
                            n:int,
                            sigma: int) -> Tuple[np.ndarray, tf.Tensor]:
    """
    Add noise to the given image n times creting n randomly smoothed versions of the image.
    Then compute the smooth gradient using the n samples and their relative model's outputs.
    Return the smooth gradient result as a tensor (np.array type) and an example smoothed
    version of the original image just for visualization

    Parameters:
    - input_ (tf.Tensor): Input image.
    - model (tf.keras.Model): A TensorFlow neural network model.
    - sigma: the noise level to apply (integer between [0,+inf) which represents a percentage)
    - n: number of samples

    Returns:
    - tf.Tensor: Modified saliency map.
    """
    rep_tensor = tf.convert_to_tensor(tf.repeat(tf.expand_dims(input_, axis=0), n, 0))
    out = tf.map_fn(lambda x: noisy_saliency(x, model, sigma), rep_tensor)
    smooth_grad = np.mean(out, 0)
    noise = tf.random.normal(input_.shape, mean=0, stddev=sigma)
    noisy_inp = input_ + noise
    return smooth_grad, noisy_inp


def plot_example_sg(ex: np.ndarray,
                    model: tf.keras.Model) -> None:
    """
    Plots the original input image, its saliency map, and the saliency map capped at the 99th percentile,
    an example of smoothed input and its corrispective

    Parameters:
    - ex (tf.Tensor): An example tensor representing an image.
    - model (tf.keras.Model): A TensorFlow neural network model.

    Returns:
    - None
    """

    saliency_map = compute_saliency(ex, model)
    smooth_saliency_map, noisy_ex = compute_smooth_saliency(ex, model, 10, 60)

    plt.rcdefaults()
    fig, ax = plt.subplots(1,6, figsize=(19,16))
    ax[0].imshow(np.array(ex, dtype=np.uint8))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Original input")
    ax[1].imshow(saliency_map, cmap="gray")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Saliency map")
    ax[2].imshow(capped_saliency(saliency_map), cmap="gray")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title("\"Capped\" saliency map")
    ax[3].imshow(normalize_to_image(noisy_ex))
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_title("Smoothed input")
    ax[4].imshow(smooth_saliency_map, cmap="gray")
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[4].set_title("Smoothed saliency map")
    ax[5].imshow(capped_saliency(smooth_saliency_map), cmap="gray")
    ax[5].set_xticks([])
    ax[5].set_yticks([])
    ax[5].set_title("\"Capped\" Smoothed map")
    plt.show()


def compare_noise_and_samples(ex: np.ndarray,
                              model: tf.keras.Model) -> None:
    """
    Plot the smooth gradient for different levels of noise and number of samples

    Parameters:
    - ex (tf.Tensor): An example tensor representing an image.
    - model (tf.keras.Model): A TensorFlow neural network model.

    Returns:
    - None
    """
    n_samples = np.arange(10, 81, 10)
    sigmas = np.arange(10, 150, 10)

    fig, ax = plt.subplots(len(n_samples), len(sigmas), figsize=(21,12), gridspec_kw={'wspace': 0, 'hspace': 0})
    ax = ax.flatten()

    for i, n in enumerate(n_samples):
      for j, s in enumerate(sigmas):
        idx =  i * len(sigmas) + j
        smooth_saliency_map, _ = compute_smooth_saliency(ex, model, s, n)
        ax[idx].imshow(smooth_saliency_map)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        if j == 0:
          ax[idx].set_ylabel(f'{n} samples')
        if i == 0:
          ax[idx].set_title(f'noise {s/255*100:.2f}%')

    plt.show()


def plot_sm_vs_sg(representative_examples: Dict[int, tf.Tensor],
                  model: tf.keras.Model,
                  n: int,
                  sigma: int) -> None:
    """
    Plots the original input image, its saliency map, and the saliency map capped at the 99th percentile,
    its smooth gradient and capped smooth gradient for given level of noise sigma and number of samples
    for each example image in the given dictionary. The keys of the dictionary are the labels and the values
    are the corresponding sampled examples

    Parameters:
    - representative_examples Dict[int, tf.Tensor]: a dictionary of example images (one for each class)
    - model (tf.keras.Model): A TensorFlow neural network model.
    - sigma: the noise level to apply (integer between [0,+inf) which represents a percentage)
    - n: number of samples

    Returns:
    - None
    """

    fig, ax = plt.subplots(23, 5, figsize=(10,50), gridspec_kw={'wspace': 0, 'hspace': 0})
    ax = ax.flatten()

    for i, ex in enumerate(representative_examples.items()):

        img = ex[1]
        label = ex[0]
        idx1 = i * 5
        idx2 = i * 5 + 1
        idx3 = i * 5 + 2
        idx4 = i * 5 + 3
        idx5 = i * 5 + 4

        saliency_map = compute_saliency(img, model)
        smooth_saliency, _ = compute_smooth_saliency(img, model, n, sigma)

        ax[idx1].imshow(np.array(img, dtype=np.uint8))
        ax[idx1].set_xticks([])
        ax[idx1].set_yticks([])
        ax[idx1].set_ylabel(f'{labels[int(label)]}')
        ax[idx2].imshow(saliency_map, cmap="gray")
        ax[idx2].set_xticks([])
        ax[idx2].set_yticks([])
        ax[idx3].imshow(capped_saliency(saliency_map), cmap="gray")
        ax[idx3].set_xticks([])
        ax[idx3].set_yticks([])
        ax[idx4].imshow(smooth_saliency, cmap="gray")
        ax[idx4].set_xticks([])
        ax[idx4].set_yticks([])
        ax[idx5].imshow(capped_saliency(smooth_saliency), cmap="gray")
        ax[idx5].set_xticks([])
        ax[idx5].set_yticks([])


        if i == 0:
          ax[idx1].set_title("Input")
          ax[idx2].set_title("Saliency Map")
          ax[idx3].set_title("Capped s.m.")
          ax[idx4].set_title("Smooth Gradient")
          ax[idx5].set_title("Capped s.g.")

    plt.show()


def parallel_saliency(dataset: tf.data.Dataset,
                      labels: Dict[int, str],
                      by_class: bool = False) -> Union[List[tf.Tensor], tf.Tensor]:
    """
    Compute the saliency map for each example in the training (or whatever) set.

    If by_class is set to True, the saliency maps are stacked in different tensors,
    divided by class, and then stored in a list. Otherwise, if by_class is set to
    False, every saliency map is stacked in a unique tensor.

    Parameters:
    - dataset (tf.data.Dataset): Input dataset containing examples.
    - labels (Dict[int, str]): Dictionary mapping class labels to their corresponding names.
    - by_class (bool): If True, stack saliency maps by class; if False, stack all maps together.

    Returns:
    - Union[List[tf.Tensor], tf.Tensor]: Stacked saliency maps, either as a list of tensors (if by_class=True)
      or as a single tensor (if by_class=False).
    """

    if by_class:
        stacked_smaps = []
        for i, l in labels.items():
            inputs = np.array([x[0] for x in dataset.unbatch().as_numpy_iterator() if x[1] == int(i)], dtype=np.float32)
            inputs = tf.convert_to_tensor(inputs)
            stacked_smaps.append(tf.map_fn(lambda x: compute_saliency(x, model), inputs))
    else:
        inputs = np.array([x[0] for x in train.unbatch().as_numpy_iterator()], dtype=np.float32)
        inputs = tf.convert_to_tensor(inputs)
        stacked_smaps = tf.map_fn(lambda x: compute_saliency(x, model), inputs)

    return stacked_smaps


def global_saliency(dataset: tf.data.Dataset,
                    labels: Dict[int, str],
                    by_class: bool = False) -> Union[List[np.ndarray], np.ndarray]:
    """
    Compute the global saliency map for a dataset.

    Parameters:
    - dataset (tf.data.Dataset): Input dataset containing examples.
    - labels (Dict[int, str]): Dictionary mapping class labels to their corresponding names.
    - by_class (bool): If True, compute global saliency maps by class; if False, compute overall global saliency map.

    Returns:
    - Union[List[np.ndarray], np.ndarray]: Global saliency map(s), either as a list of numpy arrays (if by_class=True)
      or as a single numpy array (if by_class=False).
    """

    stacked_smaps = parallel_saliency(dataset, labels, by_class)

    if by_class:
        return [np.mean(i, axis=0) for i in stacked_smaps]

    return np.mean(stacked_smaps, axis=0)


def show_global_sm(train: tf.data.Dataset,
                   labels: Dict[int, str]) -> np.ndarray:
    """
    Show global saliency maps for each class and overall dataset.

    Parameters:
    - train (tf.data.Dataset): Input training dataset containing examples.
    - labels (Dict[int, str]): Dictionary mapping class labels to their corresponding names.

    Returns:
    - np.ndarray: Overall global saliency map for the entire dataset.
    """

    fig, ax = plt.subplots(4, 6, figsize=(16, 12), gridspec_kw={'wspace': 0, 'hspace': 0})
    ax = ax.flatten()

    gs = global_saliency(train, labels, by_class=True)

    for i, saliency in enumerate(gs):
        ax[i].imshow(saliency)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f'{labels[i]}')

    glob = global_saliency(train, labels)
    ax[-1].imshow(glob)
    ax[-1].set_title("All data set", color="red")
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])

    plt.show()

    return glob


def show_weights_importance(weights: tf.Tensor,
                   labels: Dict[int, str]) -> np.ndarray:
    """
    Show the importance given by the linear model to each feature (pixel) of any
    training input (image). In this case plot the weghts for each neuron corresponding
    to a single class and also the mean of the weights w.r.t. all the classes.

    Parameters:
    - weights (tf.Tensor): The weights associated to a linear model (biases excluded)
    - labels (Dict[int, str]): Dictionary mapping class labels to their corresponding names.

    Returns:
    - None
    """
    weights = np.reshape(weights, (-1,180,180,3))

    fig, ax = plt.subplots(4, 6, figsize=(16, 12), gridspec_kw={'wspace': 0, 'hspace': 0})
    ax = ax.flatten()

    for i, w in enumerate(weights):

        importance = np.max(w, axis=-1)

        ax[i].imshow(normalize_to_image(importance))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f'{labels[i]}')

    glob_importance = np.max(np.mean(weights, axis=0), axis=2)
    ax[-1].imshow(normalize_to_image(glob_importance))
    ax[-1].set_title("All classes", color="red")
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])

    plt.show()


def step_grad(input_: tf.Tensor,
              baseline: tf.Tensor,
              model: tf.keras.Model,
              k: int,
              m: int) -> np.ndarray:
    """
    Using the baseline image and the input image, compute the intermediate result
    for one step of the approximated Integrated Gradient algorithm.

    Parameters:
    - input_ (tf.Tensor): Input image.
    - baseline (tf.Tensor): Baseline image (with a saliency map of all 0s)
    - model (tf.keras.Model): A TensorFlow neural network model.
    - k (int): k-th iteration for the approximated integrated gradient algorithm
    - m (int): total number of iterations of the approximated integrated gradient algorithm

    Returns:
    - np.array: the intermediate result for one iteration of the algorithm.
    """

    input_ = tf.expand_dims(input_, axis=0)
    baseline = tf.expand_dims(baseline, axis=0)

    with tf.GradientTape() as gt:
        gt.watch(input_)
        num = baseline + tf.cast((k/m), dtype=input_.dtype) * tf.cast((input_ - baseline), dtype=tf.float32)
        logit = model(num, training=False)
        logit = tf.squeeze(logit)
        class_score = logit[tf.argmax(logit)]

    grad = gt.gradient(class_score, input_)
    grad = tf.squeeze(grad)
    return grad



def compute_integrated(input_: tf.Tensor,
                       baseline: tf.Tensor,
                       model: tf.keras.Model,
                       m: int) -> np.ndarray:
    """
    Call m times the 'step_grad' function. Return the result og the approximated
    Integrated Gradient algorihtm.

    Parameters:
    - input_ (tf.Tensor): Input image.
    - baseline (tf.Tensor): Baseline image (with a saliency map of all 0s)
    - model (tf.keras.Model): A TensorFlow neural network model.
    - m (int): total number of iterations of the approximated integrated gradient algorithm

    Returns:
    - np.array: the intermediate result for one iteration of the algorithm.
    """

    rep_tensor = tf.convert_to_tensor(np.arange(1,m+1,1), dtype=tf.float32)

    out = np.mean(tf.map_fn(lambda x: step_grad(input_, baseline, model, x, m), rep_tensor), 0)
    coeff = np.array(input_ - baseline)
    int_grad = coeff * out
    int_grad = np.max(tf.abs(int_grad), axis=-1)
    return int_grad


def plot_sm_vs_ig(representative_examples: Dict[int, tf.Tensor],
                  baseline: tf.Tensor,
                  model: tf.keras.Model,
                  m: int,
                  n: int,
                  sigma: int) -> None:
    """
    Plots the original input image, its saliency map, and the saliency map capped at the 99th percentile,
    its smooth gradient and capped smooth gradient for given level of noise sigma and number of samples
    for each example image in the given dictionary. The keys of the dictionary are the labels and the values
    are the corresponding sampled examples

    Parameters:
    - representative_examples Dict[int, tf.Tensor]: a dictionary of example images (one for each class).
    - baseline (tf.Tensor): baseline example.
    - model (tf.keras.Model): A TensorFlow neural network model.
    - m (int): total number of iterations of the approximated integrated gradient algorithm
    - sigma: the noise level to apply (integer between [0,+inf) which represents a percentage).
    - n: number of samples

    Returns:
    - None
    """

    fig, ax = plt.subplots(23, 5, figsize=(10,50), gridspec_kw={'wspace': 0, 'hspace': 0})
    ax = ax.flatten()

    for i, ex in enumerate(representative_examples.items()):

        img = ex[1]
        label = ex[0]
        idx1 = i * 5
        idx2 = i * 5 + 1
        idx3 = i * 5 + 2
        idx4 = i * 5 + 3
        idx5 = i * 5 + 4

        smooth_saliency, _ = compute_smooth_saliency(img, model, n, sigma)
        integr_saliency = compute_integrated(img, baseline, model, m)

        ax[idx1].imshow(np.array(img, dtype=np.uint8))
        ax[idx1].set_xticks([])
        ax[idx1].set_yticks([])
        ax[idx1].set_ylabel(f'{labels[int(label)]}')
        ax[idx2].imshow(smooth_saliency, cmap="gray")
        ax[idx2].set_xticks([])
        ax[idx2].set_yticks([])
        ax[idx3].imshow(capped_saliency(smooth_saliency), cmap="gray")
        ax[idx3].set_xticks([])
        ax[idx3].set_yticks([])
        ax[idx4].imshow(integr_saliency, cmap="gray")
        ax[idx4].set_xticks([])
        ax[idx4].set_yticks([])
        ax[idx5].imshow(capped_saliency(integr_saliency), cmap="gray")
        ax[idx5].set_xticks([])
        ax[idx5].set_yticks([])


        if i == 0:
          ax[idx1].set_title("Input")
          ax[idx2].set_title("Smooth gradient")
          ax[idx3].set_title("Capped s.g.")
          ax[idx4].set_title("Integrated gradient")
          ax[idx5].set_title("Capped i.g.")

    plt.show()