from FeatureExtractor import createFE
from KitNET.KitNET import KitNET
from AnomalyDetector import AnomalyDetector
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import click
from interfaceHelpers import *


@click.group()
def main():
    pass

@main.command()
@click.option('--source', default='live', help='pcap file, tsv file or "live" to capture from interface')
@click.option('--packetnum', default=500000)
def train(source, packetnum):
    try:
        fe = createFE(source, packetnum)
        anomDetector = AnomalyDetector(trainOnNum=packetnum)
        while not anomDetector.isTrained():
            if fe.reachedPacketLimit():
                raise RuntimeError('Not enough packets for training.')
            process_packet(fe, anomDetector)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt.\n")
    finally:
        serializeModel(anomDetector)

@main.command()
@click.option('--source', default='live', help='pcap file, tsv file or "live" to capture from interface')
@click.option('--packetnum', default=500000)
@click.option('--skip', default=0, help='skip first n packets')
def run(source, packetnum, skip):
    try:
        packetnum+=skip
        modelKey = click.prompt('Please enter a key to decrypt a model')
        anomDetector = deserializeModel(modelKey)
        while not anomDetector:
            modelKey = click.prompt('Wrong key. Try again')
            anomDetector = deserializeModel(modelKey)
        fe = createFE(source, packetnum, int(skip))
        rmsesLogger = Logger(OUTPUT_PATH)
        #rmses = []
        while not fe.reachedPacketLimit():
            curVal = process_packet(fe, anomDetector)
            rmsesLogger.write(curVal)
    except KeyboardInterrupt:
        print("...writing results to file")
    #finally:
    #    rmsesLogger.write(rmses)


@main.command()
def show():
    labelsKey = click.prompt('Please enter a key to decrypt a labels file')
    labels = getLabels(labelsKey)
    while labels is None:
        labelsKey = click.prompt('Wrong key. Try again')
        labels = getLabels(labelsKey)

    plt.style.use('ggplot')
    #plt.scatter(edgecolors=None, data=labels, x='num', y='val', s=0.2)
    sns.stripplot(data=labels, x ='num', y='val', orient='h', s = 1)
    plt.show()

'''
@main.command()
def showRMSES():
    scores = pd.read_csv('rmses.csv').transpose().reset_index().reset_index()
    scores.columns = ['num', 'val']
    scores['val'] = scores['val'].apply(lambda x: float(x))
    sns.scatterplot(data = scores, x='num', y='val')
    plt.show() '''



