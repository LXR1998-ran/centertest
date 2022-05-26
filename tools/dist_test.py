import argparse
import copy
import json
import os
import sys
import torch
a=torch.Tensor([0,1])
print(a.cuda())
from det3d.torchie.trainer.trainer import example_to_device
from nuscenes.nuscenes import NuScenes
from det3d.core.utils.center_utils import (draw_umich_gaussian, gaussian_radius)
from det3d.core.bbox.box_np_ops import center_to_corner_box2d
from det3d.core.bbox.geometry import points_in_convex_polygon_jit

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--eval_det", action='store_true')

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args



def tracking():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    global voxel_size, downsample, voxel_range, num_classes, size_h, size_w
    voxel_size = np.array(cfg._voxel_size)[:2]
    downsample= cfg.assigner.out_size_factor
    voxel_range = np.array(cfg._pc_range)
    num_classes = sum([t['num_class'] for t in cfg.tasks])
    size_w, size_h = ((voxel_range[3:5] - voxel_range[:2]) / voxel_size  / downsample).astype(np.int32)
    #################checkpoint##########
    #import pdb; pdb.set_trace()    
    dataset = build_dataset(cfg.data.val)
    #################checkpoint##########
    #import pdb; pdb.set_trace()
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, args.checkpoint, map_location="cpu")
    data_loader = build_dataloader(dataset, batch_size=1, workers_per_gpu=4, dist=False,shuffle=False,)

    model = model.cuda()
    model.eval()

    detections = {}
    cpu_device = torch.device("cpu")

    prev_detections = {}
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data2/xrli/datasets/nuscenes', verbose=True)
    grids = meshgrid(size_w, size_h)

    start_id = 0
    with torch.no_grad():
        for _, data_batch in enumerate(data_loader):
            #import pdb; pdb.set_trace()
            device = torch.device(args.local_rank)
            data_batch = example_to_device(data_batch, device, non_blocking=False)
       
            prev_token = nusc.get('sample', data_batch['metadata'][0]['token'])['prev']
            track_outputs = None
            if  prev_token != '': # non-first frame
                assert prev_token in prev_detections.keys()
                box3d = prev_detections[prev_token]['box3d_global']
                box3d = (data_batch['ref_from_car'][0].detach().numpy() @ data_batch['car_from_global'][0].detach().numpy()) @ box3d
                box3d = box3d.T
                prev_detections[prev_token]['box3d_lidar'] = np.concatenate((box3d[:, :3], 
                                                            prev_detections[prev_token]['box3d_lidar'][:, 3:]), axis=1)

                prev_hm_, prev_track_id_ = render_trackmap(prev_detections[prev_token],  grids, cfg)
                prev_hm_ = prev_hm_.permute(0,2,3,1).view(1, size_h*size_w, num_classes).contiguous().to(device, non_blocking=False)
                prev_track_id_ = prev_track_id_.permute(0,2,3,1).view(1, size_h*size_w, num_classes).contiguous().to(device, non_blocking=False)

                prev_hm = []
                prev_track_id = []
                class_id = 0
                for task in cfg.tasks:
                    prev_hm.append(prev_hm_[..., class_id : class_id+task['num_class']])
                    prev_track_id.append(prev_track_id_[..., class_id : class_id+task['num_class']])
                    class_id += task['num_class']

                preds = model(data_batch, return_loss=False, return_feature=True)
                # predict tracking
                outputs, track_outputs = model.bbox_head.predict_tracking(data_batch, preds, model.test_cfg, prev_hm = prev_hm, prev_track_id=prev_track_id, new_only=False)
                outputs[0]['tracking_id'] = torch.arange(start_id, start_id+outputs[0]['scores'].size(0)).int()
                start_id += outputs[0]['scores'].size(0) 

            else: # first frame
                #import pdb; pdb.set_trace()
                outputs = model(data_batch, return_loss=False) # use predict()
                
                outputs[0]['tracking_id'] = torch.arange(start_id, start_id+outputs[0]['scores'].size(0)).int()
                start_id += outputs[0]['scores'].size(0)
            
            output = copy.deepcopy(outputs[0])
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in ["metadata"]:
                    if track_outputs is not None:
                        output[k] = torch.cat([v.clone().to(cpu_device), track_outputs[0][k].clone().to(cpu_device)], dim=0)
                    else:
                        output[k] = v.clone().to(cpu_device)
            #update the tracked objects to their current locations
            detections.update({token: output,})
            
            #import pdb; pdb.set_trace()
            prev_output = {}
            box3d_lidar = output['box3d_lidar'].clone().detach().cpu().numpy()
            box3d = np.concatenate((box3d_lidar[:, :3], np.ones((box3d_lidar.shape[0],1))), axis=1).T
            box3d = (np.linalg.inv(data_batch['car_from_global'][0]) @ np.linalg.inv(data_batch['ref_from_car'][0])) @ box3d
            prev_output['box3d_lidar'] = box3d_lidar
            prev_output['box3d_global'] = box3d
            prev_output['label_preds'] = output['label_preds'].cpu().numpy()
            prev_output['scores'] = output['scores'].cpu().numpy()
            prev_output['tracking_id'] = output['tracking_id'].cpu().numpy()
            prev_detections[output['metadata']['token']] = prev_output


    synchronize()

    all_predictions = all_gather(detections)

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    
    if args.eval_det:
        result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=False)
        if result_dict is not None:
            for k, v in result_dict["results"].items():
                print(f"Evaluation {k}: {v}")
    
    # eval tracking
    dataset.evaluation_tracking(copy.deepcopy(predictions), output_dir=args.work_dir, testset=False)

def meshgrid(w, h):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = ww.reshape(-1) 
    hh = hh.reshape(-1)

    return np.stack([ww, hh], axis=1)


def render_trackmap(preds_dicts, grids, cfg):
    prev_hm = np.zeros((1, num_classes, size_h, size_w),dtype=np.float32)
    prev_tracking_map = np.zeros((1, num_classes, size_h, size_w), dtype=np.int64) - 1
    label_preds = preds_dicts['label_preds']
    box3d_lidar = preds_dicts['box3d_lidar']
    scores = preds_dicts['scores']
    tracking_ids = preds_dicts['tracking_id']
    
    box_corners = center_to_corner_box2d(box3d_lidar[:, :2], box3d_lidar[:, 3:5], box3d_lidar[:, -1])
    box_corners = (box_corners - voxel_range[:2].reshape(1, 1, 2)) / voxel_size[:2].reshape(1, 1, 2) / downsample
    masks = points_in_convex_polygon_jit(grids, box_corners)
    
    for obj in range(label_preds.shape[0]):
        cls_id = label_preds[obj]
        score = scores[obj]
        tracking_id = tracking_ids[obj]
        size_x, size_y = box3d_lidar[obj, 3] / voxel_size[0] / downsample, box3d_lidar[obj, 4] / voxel_size[1] / downsample
        if size_x > 0 and size_y > 0:
            radius = gaussian_radius((size_y, size_x), min_overlap=0.1)
            radius = min(cfg.assigner.min_radius, int(radius))

            coor_x = (box3d_lidar[obj, 0] - voxel_range[0]) / voxel_size[0] / downsample
            coor_y = (box3d_lidar[obj, 1] - voxel_range[1]) / voxel_size[1] / downsample
            ct = np.array([coor_x, coor_y], dtype=np.float32)  
            ct_int = ct.astype(np.int32)
            # throw out not in range objects to avoid out of array area when creating the heatmap
            if not (0 <= ct_int[0] < size_w and 0 <= ct_int[1] < size_h):
                continue 
            # render center map as in centertrack
            draw_umich_gaussian(prev_hm[0, cls_id], ct, radius, score)  #

            # tracking ID map
            mask = masks[:, obj].nonzero()[0]
            coord_in_box = grids[mask, :]
            mask1 = prev_tracking_map[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] == -1
            mask2 = prev_hm[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] < score
            mask = mask[np.logical_or(mask1, mask2)]
            coord_in_box = grids[mask, :]
            prev_tracking_map[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] = tracking_id
            prev_tracking_map[0, cls_id][ct_int[1], ct_int[0]] = tracking_id
           
    return torch.from_numpy(prev_hm), torch.from_numpy(prev_tracking_map)


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
            if args.local_rank == 0:
                prog_bar.update()

    synchronize()

    all_predictions = all_gather(detections)

    print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(predictions, args.work_dir)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
   # main()
   tracking()
