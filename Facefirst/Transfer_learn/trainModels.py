import argparse
from train.transferLearning import transferLearn

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--arch", "-a", type=str, default="resnet18", help="Model architecture")
	parser.add_argument("--label", "-l", type=str, default="test", help="Model label")
	parser.add_argument("--sampling", "-s", type=str, default=None, help="Data balancing")
	parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset path")
	args = parser.parse_args()

	print(args.arch, args.label)
	transferLearn(args.arch, args.label, args.sampling, args.dataset)
