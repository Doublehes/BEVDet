savepath=./vis_dal
config=configs/dal/dal-base.py
checkpoint=./dal-base.pth
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json --root_path /datasets/nuScenese/ \
    --draw-gt  --format image