import os
from abc import abstractmethod
from typing import Dict, Tuple

from pipeline.storage import ExperimentsStorage, ModelParts


class ExternalStorage:
    def __init__(self, exps_storage: ExperimentsStorage):
        self.exps_storage = exps_storage

    @abstractmethod
    def _import_model_part(self, model_name: str, part: ModelParts, to_filepath: str) -> None:
        pass

    def import_model_part(self, model_name: str, part: ModelParts) -> None:
        """
        Import from the external storage
        """
        model_dir = self.exps_storage.get_model_dir(model_name)
        to_filepath = model_dir.get_part_filepath(part)
        self._import_model_part(model_name, part, to_filepath)

    def import_model(self, model_name: str) -> None:
        """
        Import from the external storage
        """
        for part in ModelParts:
            try:
                self.import_model_part(model_name, part)
            except:
                pass

    @abstractmethod
    def _export_model_part(self, model_name: str, part: ModelParts, from_filepath: str) -> None:
        """
        Export to the external storage
        """
        pass

    def export_model_part(self, model_name: str, part: ModelParts) -> None:
        model_dir = self.exps_storage.get_model_dir(model_name)
        from_filepath = model_dir.get_part_filepath(part)
        if os.path.exists(from_filepath):
            self._export_model_part(model_name, part, from_filepath)

    def export_model(self, model_name: str) -> None:
        for part in ModelParts:
            self.export_model_part(model_name, part)

    def list_available_models(self) -> Dict[str, Tuple[ModelParts]]:
        """
        {
            model_name: [parts]
        }
        """
        pass
