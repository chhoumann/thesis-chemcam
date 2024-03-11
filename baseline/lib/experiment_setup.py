
import mlflow
import datetime

from lib.full_flow_dataloader import load_train_test_data

class Experiment:
	def __init__(self, name: str, norm: int):
		self.name = name
		self.norm = norm
		self.X_train, self.y_train, self.X_test, self.y_test = load_train_test_data(norm=norm)


	def run(self, func):
		mlflow.set_experiment(f'{self.name}_Norm{self.norm}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

		for target in self.y_train.columns:
			with mlflow.start_run(run_name=f"{self.name}_{target}"):
				func(self.X_train, self.X_test, self.y_train[target], self.y_test[target], target, self.norm)