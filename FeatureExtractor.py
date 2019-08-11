import os
#Import dependencies
import netStat as ns
import csv
import numpy as np
from scapy.all import *
import os.path
import platform
import subprocess
import time

from PacketSniffer.sniffer import Sniffer


def createFE(file_path, limit=np.inf, skip=0):
    if file_path=="live":
        return InterfaceFE(file_path, limit, skip)

    _tshark = get_tshark_path()
    ### Find file: ###
    if not os.path.isfile(file_path):  # file does not exist
        print("File: " + file_path + " does not exist")
        raise Exception()

    ### check file type ###
    type = file_path.split('.')[-1]

    ##If file is TSV (pre-parsed by wireshark script)
    if type == "tsv":
        return TsvFE(file_path, limit, skip)

    ##If file is pcap
    elif type == "pcap" or type == 'pcapng':
        # Try parsing via tshark dll of wireshark (faster)
        if os.path.isfile(_tshark):
            pcap2tsv_with_tshark(file_path)  # creates local tsv file
            file_path += ".tsv"
            return TsvFE(file_path, limit, skip)
        # Otherwise, parse with scapy (slower)
        print("tshark not found. Trying scapy...")
        return ScapyFE(file_path, limit, skip)
    else:
        print("File: " + file_path + " is not a tsv or pcap file")
        raise Exception()



class FE():

    def __init__(self, file_path, limit, skip):
        self.path = file_path
        self.limit = limit
        self.curPacketIndx = skip

        ### Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)

    def get_next_vector(self):
        ### Extract Features
        pass

    def reachedPacketLimit(self):
        return self.curPacketIndx >= self.limit;


    def tryNSUpdate(self, IPtype, srcMAC, dstMAC, srcIP, srcport, dstIP, dstport, framelen):
        #try:
        #print(srcMAC, srcIP, srcport)
        return self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, str(srcport), dstIP, str(dstport),
                                                 int(framelen), time.time())
        #except Exception as e:
        #    print("Error in 72 line of Feature Extractor.py ", e.with_traceback())
        #    return []

    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())

class TsvFE(FE):
    def __init__(self, file_path, limit, skip):
        FE.__init__(self, file_path, limit, skip)
        maxInt = sys.maxsize
        decrement = True
        while decrement:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            decrement = False
            try:
                csv.field_size_limit(maxInt)
            except OverflowError:
                maxInt = int(maxInt / 10)
                decrement = True

        print("counting lines in file...")
        num_lines = sum(1 for line in open(self.path))
        print("There are " + str(num_lines) + " Packets.")
        self.limit = min(self.limit, num_lines - 1)
        self.tsvinf = open(self.path, 'rt', encoding="utf8")
        self.tsvin = csv.reader(self.tsvinf, delimiter='\t')
        row = self.tsvin.__next__()  # move iterator past header
        for i in range(0, skip):
            row = self.tsvin.__next__()

    def get_next_vector(self):
        if self.curPacketIndx == self.limit:
            self.tsvinf.close()
            return []

        row = self.tsvin.__next__()
        IPtype = np.nan
        timestamp = row[0]
        framelen = row[1]
        srcIP = ''
        dstIP = ''
        if row[4] != '':  # IPv4
            srcIP = row[4]
            dstIP = row[5]
            IPtype = 0
        elif row[17] != '':  # ipv6
            srcIP = row[17]
            dstIP = row[18]
            IPtype = 1
        srcport = row[6] + row[
            8]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
        dstport = row[7] + row[9]  # UDP or TCP port
        srcMAC = row[2]
        dstMAC = row[3]
        if srcport == '':  # it's a L2/L1 level protocol
            if row[12] != '':  # is ARP
                srcport = 'arp'
                dstport = 'arp'
                srcIP = row[14]  # src IP (ARP)
                dstIP = row[16]  # dst IP (ARP)
                IPtype = 0
            elif row[10] != '':  # is ICMP
                srcport = 'icmp'
                dstport = 'icmp'
                IPtype = 0
            elif srcIP + srcport + dstIP + dstport == '':  # some other protocol
                srcIP = row[2]  # src MAC
                dstIP = row[3]  # dst MAC

        self.curPacketIndx = self.curPacketIndx + 1
        return super().tryNSUpdate(IPtype, srcMAC, dstMAC, srcIP, srcport, dstIP, dstport, framelen)




class ScapyFE(FE):
    def __init__(self, file_path, limit):
        FE.__init__(self, file_path, limit, skip)
        print("Reading PCAP file via Scapy...")
        self.scapyin = rdpcap(self.path)
        self.limit = len(self.scapyin)
        print("Loaded " + str(len(self.scapyin)) + " Packets.")

    def get_next_vector(self):
        if self.curPacketIndx == self.limit:
            return []

        packet = self.scapyin[self.curPacketIndx]
        IPtype = np.nan
        timestamp = packet.time
        framelen = len(packet)
        if packet.haslayer(IP):  # IPv4
            srcIP = packet[IP].src
            dstIP = packet[IP].dst
            IPtype = 0
        elif packet.haslayer(IPv6):  # ipv6
            srcIP = packet[IPv6].src
            dstIP = packet[IPv6].dst
            IPtype = 1
        else:
            srcIP = ''
            dstIP = ''

        if packet.haslayer(TCP):
            srcport = str(packet[TCP].sport)
            dstport = str(packet[TCP].dport)
        elif packet.haslayer(UDP):
            srcport = str(packet[UDP].sport)
            dstport = str(packet[UDP].dport)
        else:
            srcport = ''
            dstport = ''

        srcMAC = packet.src
        dstMAC = packet.dst
        if srcport == '':  # it's a L2/L1 level protocol
            if packet.haslayer(ARP):  # is ARP
                srcport = 'arp'
                dstport = 'arp'
                srcIP = packet[ARP].psrc  # src IP (ARP)
                dstIP = packet[ARP].pdst  # dst IP (ARP)
                IPtype = 0
            elif packet.haslayer(ICMP):  # is ICMP
                srcport = 'icmp'
                dstport = 'icmp'
                IPtype = 0
            elif srcIP + srcport + dstIP + dstport == '':  # some other protocol
                srcIP = packet.src  # src MAC
                dstIP = packet.dst  # dst MAC

        self.curPacketIndx = self.curPacketIndx + 1
        return super().tryNSUpdate(IPtype, srcMAC, dstMAC, srcIP, srcport, dstIP, dstport, framelen)

class InterfaceFE(FE):
    def __init__(self, file_path, limit, skip):
        self.Sniffer = Sniffer()
        FE.__init__(self, file_path, limit, skip)

    def get_next_vector(self):
        if self.curPacketIndx == self.limit:
            return []
        self.curPacketIndx = self.curPacketIndx + 1
        data = self.Sniffer.getPacket()
        if data:
            return super().tryNSUpdate(*data)
        return [];



# helpers
def get_tshark_path():
    if platform.system() == 'Windows':
        return 'C:\Program Files\Wireshark\\tshark.exe'
    else:
        system_path = os.environ['PATH']
        for path in system_path.split(os.pathsep):
            filename = os.path.join(path, 'tshark')
            if os.path.isfile(filename):
                return filename
    return ''


_tshark = get_tshark_path()

def pcap2tsv_with_tshark(path):
    print('Parsing with tshark...')
    fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
    cmd =  '"' + _tshark + '" -r '+ path +' -T fields '+ fields +' -E header=y -E occurrence=f > '+ path+".tsv"
    subprocess.call(cmd,shell=True)
    print("tshark parsing complete. File saved as: "+path +".tsv")