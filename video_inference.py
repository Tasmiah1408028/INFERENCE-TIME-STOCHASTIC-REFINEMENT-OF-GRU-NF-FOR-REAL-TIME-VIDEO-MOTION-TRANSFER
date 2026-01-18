import os, sys
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from FOMM.Source_Model.logger import Logger, Visualizer
import numpy as np
import imageio
from FOMM.Source_Model.sync_batchnorm import DataParallelWithCallback
from FOMM.Source_Model.modules.RNN_prediction_module import PredictionModule
from FOMM.Source_Model.augmentation import SelectRandomFrames, SelectFirstFrames_two, VideoToTensor
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from FOMM.Source_Model.frames_dataset import FramesDataset
import tensorflow.compat.v1 as tf
import pickle
import gc
import yaml
from FOMM.Source_Model.modules.generator import OcclusionAwareGenerator
from FOMM.Source_Model.modules.keypoint_detector import KPDetector
from FOMM.Source_Model.logger import Logger, Visualizer, Visualizer_slow
from torch import nn
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


inference_index_st = int(os.environ["inference_index_start"])
inference_index_end = int(os.environ["inference_index_end"])

with open("GRU-SNF_vox8-16_test_video_unstd_list_100_mcmc.pkl", "rb") as f:
    test_videos_unstd_list = pickle.load(f)

print("Loaded list type:", type(test_videos_unstd_list))

####### call the config functions and inference dataloader #########
config="config/abs-vox.yml"
frames = 30
# Test dataset
with open(config) as f:
    config = yaml.safe_load(f)


full_dataset = FramesDataset(is_train=(False), **config['dataset_params'],mode="RNN") # test

### call the functions
occ_generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params']).to(device)
kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params']).to(device)

log_dir="log/test-reconstruction-vox"
checkpoint="FOMM/Trained_Models/vox-cpk.pth.tar"

if checkpoint is not None:
    Logger.load_cpk(checkpoint, generator=occ_generator, kp_detector=kp_detector)
else:
    raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")

def save_obj(obj, name ):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

png_dir = os.path.join(log_dir, 'diversity/png')
log_dir = os.path.join(log_dir, 'vox_8-16_GRU-SNF_recon_diversity_100') 

if checkpoint is not None:
    Logger.load_cpk(checkpoint, generator=occ_generator, kp_detector=kp_detector)
else:
    raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")

# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

# if not os.path.exists(png_dir):
#     os.makedirs(png_dir)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

if torch.cuda.is_available():
    generator = DataParallelWithCallback(occ_generator)
    kp_detector = DataParallelWithCallback(kp_detector)

generator.eval()
kp_detector.eval()

prediction_params = config['prediction_params']

num_epochs = prediction_params['num_epochs']
lr = prediction_params['lr']
bs = prediction_params['batch_size']
num_frames = prediction_params['num_frames']
# loss_list_total = []
# fvd_list_total = []

dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=1)

for it, x in tqdm(enumerate(dataloader)):
# for it in range(inference_index_st, inference_index_end):
    # dataset = full_dataset.__getitem__(it)
    if it >= inference_index_st and it < inference_index_end:
        if config['reconstruction_params']['num_videos'] is not None:
            for index, driving_vid in enumerate(test_videos_unstd_list[it]):
                if it > config['reconstruction_params']['num_videos']:
                    break
                with torch.no_grad():
                    predictions = []
                    visualizations = []

                    ######## keypoints ########
                    kp_driving_video = driving_vid.reshape(-1,10,6) #here the test_videos_unstd_list is used
                    kp_driving_video = torch.tensor(kp_driving_video).to(device)
                    kp_source = {"value":kp_driving_video[0,:,:2].reshape(1,10,2).to(device),"jacobian":kp_driving_video[0,:,2:].reshape(1,10,2,2).to(device)} # kp of the ith frame
                    generator_st = generator.float().to(device)
                ##### Start generator
                # loss_list = []
                # fvd_list = []
                for i in tqdm(range(((x['video'].shape[2])//frames)*frames)): # cut the last <24 frames
                    source = x['video'][:, :, 0].to(device)
                    driving = x['video'][:, :, i].to(device)
                    kp_driving = {"value":kp_driving_video[i,:,:2],"jacobian":kp_driving_video[i,:,2:]} # kp of the ith frame
                    kp_driving['value'] = kp_driving['value'].reshape(1,10,2)
                    kp_driving['jacobian'] = kp_driving['jacobian'].reshape(1,10,2,2)
                    kp_driving['value'] = kp_driving['value'].float()
                    kp_driving['jacobian'] = kp_driving['jacobian'].float()
                    source = source.float()
                    # Convert all tensors in kp_source dictionary to float
                    kp_source = {key: value.float() for key, value in kp_source.items()}
                    out = generator_st(source, kp_source=kp_source, kp_driving=kp_driving)

                    out['kp_source'] = kp_source
                    out['kp_driving'] = kp_driving
                    del out['sparse_deformed']
                    predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                    visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                            driving=driving, out=out)
                    visualizations.append(visualization)
                    del driving, source, kp_driving, out, visualization
                    gc.collect()
                    torch.cuda.empty_cache()
                    # mse loss
                    # if np.abs(out['prediction'].detach().cpu().numpy() - driving.cpu().numpy()).mean() != 0:
                    #     loss_list.append(np.abs(out['prediction'].detach().cpu().numpy() - driving.cpu().numpy()).mean())
                    #     # Calculate FVD for each frame using ground truth and predicted videos
                    #     ground_truth_features = driving.detach().cpu().permute(0,2,3,1).reshape(256,256,3)
                    #     predicted_features = out['prediction'].detach().cpu().permute(0,2,3,1).reshape(256,256,3)
                    #     fvd_list.append(compute_fvd(ground_truth_features, predicted_features))
                
                predictions = np.concatenate(predictions, axis=1)
                imageio.imsave(os.path.join(png_dir, x['name'][0] + "_" + str(index) + '.png'), (255 * predictions).astype(np.uint8))
                image_name = x['name'][0]+ f"-{str(index)}" + config['reconstruction_params']['format']
                imageio.mimsave(os.path.join(log_dir, f"{image_name}"), visualizations)
                del predictions, visualizations
                gc.collect()
                torch.cuda.empty_cache()

#             print("Reconstruction loss: %s" % np.mean(loss_list))
#             loss_list_total.append(np.mean(loss_list))

#             print("FVD Score: %s" % np.mean(fvd_list))
#             fvd_list_total.append(np.mean(fvd_list))


# print("mean Reconstruction loss: %s" % np.mean(loss_list_total))
# print("mean FVD score: %s" % np.mean(fvd_list_total))
