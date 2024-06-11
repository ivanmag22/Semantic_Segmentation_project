# Semantic_Segmentation_project
**Title**: Efficient Domain Adaptation for Real-Time Semantic Segmentation with Lightweight Networks and Discriminators

For "Advanced Machine Learning" course at Politecnico di Torino 

Made by:
- Ivan Magistro Contenta
- Yalda Sadat Mobargha
- Luca Sturaro

The repository contains:
- *model/*: definitions of models and trained models
    - **STDC-net**: *model_stages.py* and *stdcnet.py*
    - **BiSeNet v1**: *bisenetv1.py*
    - *best_models/* contains trained models on Domain Shift and Domain Adaptation tasks
        - *Domain Shift*
            - without data augmentation: *p2c_lr_0001_bs_8_notaug_Saved_model_epoch_50.pth*
            - with data augmentation: *p2c_lr_0001_bs_8_aug_Best_model_epoch_35.pth*
        - *Domain Adaptation*
            - **STDC-net**: *p3_lr_0001_lrD_00001_bs_8_Saved_model_epoch_50.pth*
            - **BiSeNet v1**: *p4_bisenetv1_domadpt_lightdiscr_BSv1_Best_model_epoch_25.pth*
- *notebook_files/*: implementation of training, validation and other techniques
    - *run_stdc_bisenetv1.ipynb*: useful to run different tasks of the code on *GPU* (Colab)
    - *cpu_execution.ipynb*: useful to run different tasks of the code on *CPU* (Colab)
    - *metrics.ipynb*: it contains the metrics of different networks and discriminators, but also the outputs of the best models to be compared to the images and ground truths