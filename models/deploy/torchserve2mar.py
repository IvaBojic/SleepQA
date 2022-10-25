# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils
except ImportError:
    package_model = None


def mmdet2torchserve(
        src_root: str,
        output_folder: str,
        model_name: str,
        model_version: str = '1.0',
        force: bool = False,
):
    """Converts MMDetection model (config + checkpoint) to TorchServe `.mar`.
    """
    dummy_file = "README.md"
    # check folder existence
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if os.path.isfile("./qa_pipeline/qa_pipeline.mar"):
        os.remove("./qa_pipeline/qa_pipeline.mar")
    with TemporaryDirectory() as tmpdir:
        print(os.listdir(src_root))
        args = Namespace(
            **{
                'model_file': dummy_file,
                'handler': 'end2end_api.py',
                'model_name': model_name,
                'version': model_version,
                'export_path': output_folder,
                "serialized_file": dummy_file,
                'requirements_file': dummy_file,
                "extra_files": "./",
                'force': force,
                'runtime': 'python',
                'archive_format': 'default'
            })
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)


if __name__ == '__main__':
    mmdet2torchserve("./", "./qa_pipeline/", "qa_pipeline", force=True)