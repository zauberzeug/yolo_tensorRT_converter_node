from yolo_tensorRT_converter import YoloToTensorRTConverter
import uvicorn
from learning_loop_node.converter.converter_node import ConverterNode
import os
import logging

logging.basicConfig(level=logging.INFO)

yolo_tensorRT_converter = YoloToTensorRTConverter(source_format='yolo', target_format='tensorrt')
node = ConverterNode(uuid='85ef1a58-308d-4c80-8931-43d1f752f4f3', name='test', converter=yolo_tensorRT_converter)

if __name__ == "__main__":
    reload_dirs = ['./restart'] if os.environ.get('MANUAL_RESTART', None) else ['./', './learning-loop-node']

    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True, reload_dirs=reload_dirs)
