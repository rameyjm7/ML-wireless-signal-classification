import importlib

class GenericModulationClassifier:
    def __init__(self, model_name, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        self.model_name = model_name
        self.data_path = data_path
        self.model_path = model_path
        self.stats_path = stats_path
        self.classifier = self.load_classifier()

    def load_classifier(self):
        try:
            # Construct module and class name based on model_name
            module_name = f"ml_wireless_classification.{self.model_name}"
            class_name = "ModulationLSTMClassifier"

            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Get the classifier class from the module
            classifier_class = getattr(module, class_name)

            # Instantiate and return the classifier
            return classifier_class(data_path=self.data_path, model_path=self.model_path, stats_path=self.stats_path)
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")
            raise
        except AttributeError:
            print(f"Class {class_name} not found in module {module_name}.")
            raise

    def get_classifier(self):
        return self.classifier

