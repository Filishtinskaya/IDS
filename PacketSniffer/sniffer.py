import socket
from .trafficClasses import Ethernet, IPv4, ICMP, TCP, UDP, HTTP, ARP
class Sniffer:
    def __init__(self):
        self.conn = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))

    def getPacket(self):
        raw_data, addr = self.conn.recvfrom(65535)

        eth = Ethernet(raw_data)
        srcport = None
        dstport = None
        ipsrc = None
        ipdst = None

        if eth.proto == 0x0800:
            ipv4 = IPv4(eth.data)
            ipsrc = ipv4.src
            ipdst = ipv4.target

            # ICMP
            if ipv4.proto == 1:
                srcport=dstport='icmp'
            # TCP
            elif ipv4.proto == 6:
                tcp = TCP(ipv4.data)
                srcport=tcp.src_port
                dstport=tcp.dest_port
            # UDP
            elif ipv4.proto == 17:
                udp = UDP(ipv4.data)
                srcport=udp.src_port
                dstport=udp.dest_port

            return (4, eth.src_mac, eth.dest_mac, ipv4.src, srcport, ipv4.target, dstport, len(raw_data))
        elif eth.proto == int('0x0806', 16):
            arp = ARP(eth.data)
            return (4, arp.srcMAC, arp.targetMAC, arp.srcIP, 'arp', arp.targetIP, 'arp', len(raw_data))
        elif eth.proto == int('0x86dd', 16):
            print('ip v6 not supported')
            return None
        else:
            print("Unknown protocol ", hex(eth.proto))
            return None


