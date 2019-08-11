from Kitsune import Kitsune
import numpy as np
import time
import seaborn as sns

packet_limit = 20000 #the number of packets to process

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 700 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 10000 #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune


RMSEs = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    i+=1
    rmse = K.proc_next_packet()
    if i % 1000 == 0:
        print(i)
    if rmse == -1:
        break

    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

#print(RMSEs)
#sns.distplot(np.array(RMSEs))


# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
from scipy.stats import norm
benignSample = np.log(RMSEs[FMgrace+ADgrace+1:100000])
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

# plot the RMSE anomaly scores
print("Plotting results")
from matplotlib import pyplot as plt
from matplotlib import cm
plt.figure(figsize=(10,5))
fig = plt.scatter(range(FMgrace+ADgrace+1,len(RMSEs)),RMSEs[FMgrace+ADgrace+1:],s=1.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
plt.yscale("log")
plt.title("Anomaly Scores from Kitsune's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Packets")
#plt.annotate('Mirai C&C channel opened [Telnet]', xy=(121662,RMSEs[121662]), xytext=(151662,1),arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.annotate('Mirai Bot Activated\nMirai scans network\nfor vulnerable devices', xy=(122662,10), xytext=(122662,150),arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.annotate('Mirai Bot launches DoS attack', xy=(370000,100), xytext=(390000,1000),arrowprops=dict(facecolor='black', shrink=0.05),)
figbar=plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()





