import shutil
from typing import List
from learning_loop_node.converter.converter import Converter
from learning_loop_node.converter.model_information import ModelInformation
import subprocess
import shutil
from glob import glob
import os
import re
import json
from fastapi.encoders import jsonable_encoder


class YoloToTensorRTConverter(Converter):

    async def _convert(self, model_information: ModelInformation) -> None:
        shutil.move(glob(f'{self.model_folder}/*.weights')
                    [0], f'{self.model_folder}/model.weights')

        if not os.path.exists(f'{self.model_folder}/training.cfg'):
            raise Exception('training.cfg missing')
        if not os.path.exists(f'{self.model_folder}/model.weights'):
            raise Exception('model.weights missing')
        if not os.path.exists(f'{self.model_folder}/names.txt'):
            raise Exception('names.txt missing')

        shutil.rmtree('/tkDNN/darknet_fp32.rt', ignore_errors=True)
        shutil.rmtree('/tkDNN/darknet/layers', ignore_errors=True)
        os.makedirs('/tkDNN/darknet/layers')

        shutil.rmtree('/model', ignore_errors=True)
        shutil.copytree(self.model_folder, '/model')

        metadata = self.parse_meta_data(model_information)
        with open('/model/model.json', 'w') as f:
            json.dump(jsonable_encoder(metadata), f)

        cmd = f'export TKDNN_MODE=FP32 && cd /tkDNN/darknet && ./darknet export /model/training.cfg /model/model.weights layers'
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
        if p.returncode != 0:
            raise Exception(f'could not convert model. Command was : {cmd}')

        cmd = 'cd /tkDNN/build && ./test_yolo4tiny'
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

        shutil.move('/tkDNN/darknet_fp32.rt', '/model/model.rt')

    def parse_meta_data(self, model_information: ModelInformation):
        with open('/model/names.txt') as f:
            names = [line.rstrip('\n') for line in f.readlines()]
        categories = [
            category.__dict__ for category in model_information.project_categories if category.name in names]
        assert len(categories) == len(
            names), f'could not match names {names} with categories {model_information.project_categories}'
        width, height = self.read_width_and_height()
        return {
            'host': 'https://learning-loop.ai',
            'organization': model_information.organization,
            'project': model_information.project,

            'version': model_information.version,
            'id': model_information.model_id,
            'format': self.target_format,
            'categories': categories,
            'resolution': width,
        }

    def get_converted_files(self, model_id: str) -> List[str]:
        return ['/model/model.rt', '/model/model.json']

    # copied from yolo_cfg_helper.py

    def read_width_and_height(self):
        with open('/model/training.cfg', 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("width="):
                width = re.findall(r'\d+', line)[0]

            if line.startswith("height="):
                height = re.findall(r'\d+', line)[0]
        return width, height
