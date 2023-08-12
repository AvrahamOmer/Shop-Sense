#!/bin/bash
set -e


python object_trarcking.py -g -c -m \
    --source metrics/metrics_data_videos/front_1.mp4,metrics/metrics_data_videos/store_1.mp4 \
    --destination metrics/metrics_result_videos/track_front_1.mp4,metrics/metrics_result_videos/track_store_1.mp4