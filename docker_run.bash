docker run --gpus all \
  --shm-size=16g \
  -v /home/double/Documents/BEVDet:/BEVDet \
  -v /home/double/Documents/BEVDet/data:/datasets \
  -it bevdet:0.8 bash
