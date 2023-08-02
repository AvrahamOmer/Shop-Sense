#!/bin/bash
set -e


python object_trarcking.py -g -c -m \
    --source metrics/metrics_data_videos/front_3.mp4,metrics/metrics_data_videos/store_3.mp4 \
    --destination metrics/metrics_result_videos/track_front_3.mp4,metrics/metrics_result_videos/track_store_3.mp4