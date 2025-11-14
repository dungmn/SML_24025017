import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
#Plot results
# np.save('jaggArchs',jaggArchs)
Sizes = np.arange(0.1,2.1,0.3)
archs = ['vgg16','resnet50','densenet201']

jaggArchs = np.load('jaggArchs.npy', allow_pickle=True)
plt.rcParams.update({'font.size': 8})
plt.rcParams['xtick.labelsize']=4
plt.rcParams['ytick.labelsize']=4
plt.close('all')
# plt.figure(figsize=(10,10))
ax = plt.subplot(311)
ax.plot(Sizes,jaggArchs[0,0,:,0],'o-r',markersize=4,alpha=0.7)
ax.plot(Sizes,jaggArchs[1,0,:,0],'o-g',markersize=4,alpha=0.7)
ax.plot(Sizes,jaggArchs[2,0,:,0],'o-b',markersize=4,alpha=0.7)
plt.xticks(Sizes)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xlabel('Crop size (of the original size)')
plt.ylabel('P(failure)')
plt.tight_layout()
# plt.legend(archs)



ax = plt.subplot(312)
ax.plot(Sizes,jaggArchs[0,0,:,1],'o-r',markersize=4,alpha=0.7)
ax.plot(Sizes,jaggArchs[1,0,:,1],'o-g',markersize=4,alpha=0.7)
ax.plot(Sizes,jaggArchs[2,0,:,1],'o-b',markersize=4,alpha=0.7)
plt.xticks(Sizes)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xlabel('Crop size (of the original size)')
plt.ylabel('Mean absolute change')
plt.tight_layout()
# plt.legend(archs)

ax = plt.subplot(313)
ax.plot(Sizes,jaggArchs[0,0,:,2],'o-r',markersize=4,alpha=0.7)
ax.plot(Sizes,jaggArchs[1,0,:,2],'o-g',markersize=4,alpha=0.7)
ax.plot(Sizes,jaggArchs[2,0,:,2],'o-b',markersize=4,alpha=0.7)
plt.xticks(Sizes)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('Crop size')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.legend(archs)
# plt.show()
plt.savefig('CheckTranslationInvariance.pdf',bbox_inches='tight')