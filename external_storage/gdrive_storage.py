"""
Class for importing and exporting models from Google Drive.
Only for local use as the library `pydrive` is not allowed in the task.
"""
import os
import shutil
import tempfile
from typing import Dict, Tuple

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from external_storage.external_storage import ExternalStorage
from pipeline.storage import ExperimentsStorage, ModelParts


ARCHIVE_FORMAT = 'zip'


def _archive_file(archive_filepath: str, filepath: str) -> str:
    """
    returns archive filepath
    """
    if os.path.isdir(filepath):
        shutil.make_archive(archive_filepath, ARCHIVE_FORMAT, root_dir=filepath)
    else:
        parent_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        shutil.make_archive(archive_filepath, ARCHIVE_FORMAT, root_dir=parent_dir, base_dir=filename)
    return archive_filepath + '.' + ARCHIVE_FORMAT


class GDriveStorage(ExternalStorage):
    PARTS_NAMES = {
        ModelParts.CONFIG: 'config',
        ModelParts.TRAINED_MODEL: 'model',
        ModelParts.TRAINING_CHECKPOINT: 'checkpoint',
    }

    CONFIG_SUBDIR_NAME = 'config_dir'

    def __init__(self, exps_storage: ExperimentsStorage, storage_dir_id: str,
                 client_secrets_filepath: str = './client_secrets.json',
                 gauth=None):
        """
        :param storage_dir_id: the id of the GDrive directory where experiments are stored
        """
        super().__init__(exps_storage)
        if gauth is None:
            GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = client_secrets_filepath
            gauth = GoogleAuth()
            url = gauth.GetAuthUrl()
            print(f'Visit the url:\n{url}')
            code = input('Enter the code')
            gauth.Auth(code)
            # gauth.CommandLineAuth()
        self.drive = GoogleDrive(gauth)
        self.storage_dir_id = storage_dir_id

    def _get_part_name(self, part: ModelParts):
        if part not in self.PARTS_NAMES:
            raise RuntimeError(f'Model part {part} is not supported by GDriveStorage')
        return self.PARTS_NAMES[part]

    def _get_subdir(self, parent_dir_id: str, subdir_name: str) -> str:
        """
        :return: id
        """
        query = f'"{parent_dir_id}" in parents and title="{subdir_name}" and trashed=false'
        files = self.drive.ListFile({'q': query}).GetList()
        if len(files) != 0:
            return files[0]['id']
        else:
            file_metadata = {
                'title': subdir_name,
                'parents': [{'id': parent_dir_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive.CreateFile(file_metadata)
            folder.Upload()
            return folder['id']

    def _get_model_dir(self, model_name: str) -> str:
        """
        Creates if it does not exist
        :return: id
        """
        return self._get_subdir(parent_dir_id=self.storage_dir_id, subdir_name=model_name)

    def _import_model_part(self, model_name: str, part: ModelParts, to_filepath: str) -> None:
        part_name = self._get_part_name(part)
        model_dir = self._get_model_dir(model_name)
        with tempfile.TemporaryDirectory() as archive_dir:
            downloaded_archive_filepath = os.path.join(archive_dir,
                                                       model_name + '.' + ARCHIVE_FORMAT)
            query = f'"{model_dir}" in parents and title="{part_name}" and trashed=false'
            files = self.drive.ListFile({'q': query}).GetList()
            if len(files) == 0:
                raise RuntimeError(f'No part {part} found')
            archive_file = files[0]
            archive_file.GetContentFile(downloaded_archive_filepath)

            if os.path.exists(to_filepath) and os.path.isdir(to_filepath):
                shutil.unpack_archive(downloaded_archive_filepath, to_filepath, ARCHIVE_FORMAT)
            else:
                to_dirpath = os.path.dirname(to_filepath)
                imported_filename = os.path.basename(to_filepath)
                shutil.unpack_archive(downloaded_archive_filepath, to_dirpath, ARCHIVE_FORMAT)
                unpacked_filepath = os.path.join(to_dirpath, os.path.basename(downloaded_archive_filepath))
                os.rename(unpacked_filepath, os.path.join(to_dirpath, imported_filename))

    def _upload_file(self, to_dir_id: str, to_filename: str, filepath: str) -> None:
        query = f"'{to_dir_id}' in parents and title='{to_filename}' and trashed=false"
        files = self.drive.ListFile({'q': query}).GetList()
        if len(files) != 0:
            for file in files:
                file.Delete()

        file = self.drive.CreateFile(
            {'title': to_filename, 'parents': [{'id': to_dir_id}]})
        file.SetContentFile(filepath)
        file.Upload()

    def _export_dir(self, to_dir_id: str, from_dirpath: str) -> None:
        """
        Without archiving, `from_dirpath` must not contain directories
        """
        for filename in os.listdir(from_dirpath):
            filepath = os.path.join(from_dirpath, filename)
            self._upload_file(to_dir_id, filename, filepath)

    def _export_as_archive(self, dir_id: str, archive_name: str, from_path: str) -> None:
        with tempfile.TemporaryDirectory() as archive_dir:
            archive_filepath = _archive_file(os.path.join(archive_dir, archive_name),
                                             filepath=from_path)
            self._upload_file(dir_id, archive_name, archive_filepath)

    def _export_model_part(self, model_name: str, part: ModelParts, from_path: str) -> None:
        part_name = self._get_part_name(part)
        model_dir = self._get_model_dir(model_name)

        # extra copy for convenience
        if part is ModelParts.CONFIG:
            config_subdir = self._get_subdir(parent_dir_id=model_dir, subdir_name=self.CONFIG_SUBDIR_NAME)
            self._export_dir(config_subdir, from_path)

        self._export_as_archive(model_dir, part_name, from_path)

    def list_available_models(self) -> Dict[str, Tuple[ModelParts]]:
        models_list = self.drive.ListFile(
            {'q': f"'{self.storage_dir_id}' in parents and trashed=false"}).GetList()

        res = {}

        for model in models_list:
            model_parts = []
            for part in ModelParts:
                query = f"'{model['id']}' in parents and title='{self._get_part_name(part)}' and trashed=false"
                files = self.drive.ListFile({'q': query}).GetList()
                if len(files) != 0:
                    model_parts.append(part)
            res[model['title']] = tuple(model_parts)

        return res

    def _import_model(self, model_name: str, to_dirpath: str) -> None:
        with tempfile.TemporaryDirectory() as archive_dir:
            downloaded_archive_filepath = os.path.join(archive_dir,
                                                       model_name + '.' + ARCHIVE_FORMAT)
            query = f'"{self.storage_dir_id}" in parents and title="{model_name}.{ARCHIVE_FORMAT}" and trashed=false'
            archive_file = self.drive.ListFile({'q': query}).GetList()[0]
            archive_file.GetContentFile(downloaded_archive_filepath)

            shutil.unpack_archive(downloaded_archive_filepath, to_dirpath, ARCHIVE_FORMAT)
