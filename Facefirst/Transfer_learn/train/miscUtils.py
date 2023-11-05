import matplotlib.pyplot as plt

def plot_graphs(history, model_arch, model_name, balancing, save=True):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
	x = range(1, history['prev_epochs']+1)
	ax1.plot(x, history['val_acc'], label="val_acc")
	ax1.plot(x, history['train_acc'], label="train_acc")
	ax1.vlines(history['best_epoch'], 0, 1, linestyles='dashed', label="best_epoch", color='k')
	ax1.legend()

	x = range(1, history['prev_epochs']+1)
	ax2.plot(x, history['val_loss'], label="val_loss")
	ax2.plot(x, history['train_loss'], label="train_loss")
	ax2.vlines(history['best_epoch'], 0, 1, linestyles='dashed', label="best_epoch", color='k')
	ax2.legend()

	if save:
		plt.savefig(f"./saved_models/{model_arch}/{model_name}_{balancing}.png")
	plt.show()
