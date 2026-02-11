savepath=./vis
config=configs/bevdet/bevdet-r50-dev.py
checkpoint=./bevdet-r50.pth
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json --root_path /datasets/nuScenese/ \
    --draw-gt  --format image