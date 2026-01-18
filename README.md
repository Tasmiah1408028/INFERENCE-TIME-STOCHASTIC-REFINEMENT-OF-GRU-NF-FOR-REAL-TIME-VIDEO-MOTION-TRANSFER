## Description of the repository:
This is a repo containing files related to keypoint prediction and video generation using GRU-NF and GRU-SNF in the First Order Motion Model (FOMM) pipeline.

GRU-NF_keypoints.ipynb is used to train the GRU-NF model and apply the trained model on test data. The resulting keypoints are saved in pkl file and video_inference.py is used to generate videos from the keypoints. GRU-SNF_keypoints.ipynb is used to apply the same trained GRU-NF model on test data and to save the resulting keypoints. Similarly, video_inference.py is used to generate videos from the keypoints.

To run each of 44 videos of test data parallely to get 100 samples from each, we ran sharded jobs. The necessary files are in this link: https://gitlab.nrp-nautilus.io/byungheon-jeong/first-order-model-distributed/-/tree/parallel/training-infra
(Change the pkl file as needed)

The script is run like this: 
python3 job_generator.py --number_of_shard 44 --path_to_ptk_file GRU-NF_vox8-16_test_video_unstd_list_100.pkl --path_to_yaml_template core_template.yaml

When the yaml is generated, execute:
<kubectl create -f job_yaml>
to start the sharded jobs.

MAE_APD.py is used to compute the MAE and APD of the generated videos. Save the results in a log file to use the file to compute APD to MAE ratio.


## Checkpoints for the FOMM model and keypoints

Checkpoints for the FOMM model trained on the Voxceleb dataset can be found under this google drive link. https://drive.google.com/drive/folders/1pachVtWHibzDi3E61jUmqFfz2hVxA1GX?usp=drive_link.

This file "vox-cpk.pth.tar" has been sourced using the link in the original FOMM github: https://github.com/AliaksandrSiarohin/first-order-model.

To run this file in the attached video_inference.py, please copy the checkpoint file to the following path "FOMM/Trained_Models/".

The keypoints corresponding to 3883 Voxceleb videos which can be used to train the GRU-NF can be found with the same google drive link.
