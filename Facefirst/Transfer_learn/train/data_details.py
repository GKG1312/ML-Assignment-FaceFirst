import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')


def get_count(labels, I2F, plot=False):
    # dataset_paths = np.array(dataset_paths)
    labels = list(labels)

    samples = {}
    for label in np.unique(labels):
        samples[I2F[int(label)]] = labels.count(label)    
    print(samples)

    if plot:
        plt.bar(list(samples.keys()), samples.values(), width=0.5, color=['red', 'orange', 'green', 'maroon'])
        # plt.title("Over Sampled Data Distribution")
        plt.ylabel("#Samples", fontsize=13)
        plt.tight_layout()
        # plt.savefig('Over_Sampled_distribution.png', bbox_inches='tight')
        plt.show()


def random_vis(img, label):
    image = []
    label = []
    class0_idx = np.where(label==0)[0]
    class1_idx = np.where(label==1)[0]
    class2_idx = np.where(label==2)[0]
    class3_idx = np.where(label==3)[0]

    sample0 = np.random.choice(class0_idx, 10, replace=False)
    sample1 = np.random.choice(class1_idx, 10, replace=False)
    sample2 = np.random.choice(class2_idx, 10, replace=False)
    sample3 = np.random.choice(class3_idx, 10, replace=False)
    

# dataset, I2F, F2I = loadDataset("./Train")
# dataset = np.array(dataset)
# get_count(dataset[:,1], I2F)
