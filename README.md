# ExaplainNET
The main purpose of this study is the implementation of various explainable AI techniques. An attempt will then be made to replicate via code the analyses reported in the suggested scientific papers.

To do so, I will use of a data set from Kaggle consisting of color (RGB) microscope [images of pollen grains](https://www.kaggle.com/datasets/andrewmvd/pollen-grain-image-classification), specifically divided into 23 classes and the main task of the neural network will be to correctly classify these images.

![sample_images](https://github.com/Engrima18/ExaplainNET/assets/93355495/679d78c3-56e3-46bd-a843-30fd042a4ee7)

## Instructions

Open the notebook in colab and follow the instructions you will find in the markdown cells at the beginning <a target="_blank" href="https://colab.research.google.com/github/Engrima18/ExaplainNET/blob/main/report.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>.

In the jupyter file are defined all the functions for the proper obtaining of the results and the deep learning models needed for the image classification task. In this case I opted for a simple CNN achieving the following performances:

![nn_performances](https://github.com/Engrima18/ExaplainNET/assets/93355495/34bb85fe-ca1d-438f-80ef-f11beaf5a049)

Please, for a more complete understanding of the results, read the colab notebook or the `report.ipynb` file in this repository.

## Main findings 

In practice, the used techniques will consist of assigning a certain weight to the features (in this case pixels) of an example in our data set based on some manipulation of the gradients in the computational graph of the network.

The first technique is the [Saliency Map](https://arxiv.org/abs/1312.6034) and its capped version for a better visualization.

![saliency_map_analysis](https://github.com/Engrima18/ExaplainNET/assets/93355495/054ad8d8-71ae-4280-bd0f-0773c9054f09)


Then I reported the more interesting [Smooth gradient](https://arxiv.org/abs/1706.03825) technique. I started from the selection of best combination of hyperparameters.

![noise_vs_samples](https://github.com/Engrima18/ExaplainNET/assets/93355495/27b15dc1-ab2e-4802-8f65-33d998beae0f)

Finally I got better results compared to the Saliency map method.

![smooth_grad_analysis](https://github.com/Engrima18/ExaplainNET/assets/93355495/26c1cf2c-4de9-4c16-a80b-6e2e35abc108)

Finally the [Integrated gradients](https://arxiv.org/abs/1703.01365) technique seems to overperfrom all the other algorithms.

![integrated_grad_analysis](https://github.com/Engrima18/ExaplainNET/assets/93355495/28ce524d-f879-4541-8109-d48cb5ff496f)


## Used technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
