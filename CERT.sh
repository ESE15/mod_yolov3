#!/bin/bash

for (( ; ; ))
do
	./darknet detector demo cfg/coco.data cfg/yolov3.cfg ~/darknet/yolov3.weights STREAMING data/obj.data data/yolov3test.cfg ./yolov3_10000.weights
done
