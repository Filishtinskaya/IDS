import socket
import struct

class Ethernet:
    def __init__(self, raw_data):

        dest, src, prototype = struct.unpack('! 6s 6s H', raw_data[:14])

        self.dest_mac = get_mac_addr(dest)
        self.src_mac = get_mac_addr(src)
        self.proto = prototype
        self.data = raw_data[14:]

class HTTP:
    def __init__(self, raw_data):
        try:
            self.data = raw_data.decode('utf-8')
        except:
            self.data = raw_data

class ICMP:
    def __init__(self, raw_data):
        self.type, self.code, self.checksum = struct.unpack('! c c H', raw_data[:4])
        self.data = raw_data[4:]


class IPv4:
    def __init__(self, raw_data):
        version_header_length = raw_data[0]
        self.version = version_header_length >> 4
        self.header_length = (version_header_length & 15) * 4
        self.ttl, self.proto, src, target = struct.unpack('! 8x B B 2x 4s 4s', raw_data[:20])
        self.src = socket.inet_ntoa(src)
        self.target = socket.inet_ntoa(target)
        self.data = raw_data[self.header_length:]


class TCP:
    def __init__(self, raw_data):
        (self.src_port, self.dest_port, self.sequence, self.acknowledgment, offset_reserved_flags) = struct.unpack(
            '! H H L L H', raw_data[:14])
        offset = (offset_reserved_flags >> 12) * 4
        self.flag_urg = (offset_reserved_flags & 32) >> 5
        self.flag_ack = (offset_reserved_flags & 16) >> 4
        self.flag_psh = (offset_reserved_flags & 8) >> 3
        self.flag_rst = (offset_reserved_flags & 4) >> 2
        self.flag_syn = (offset_reserved_flags & 2) >> 1
        self.flag_fin = offset_reserved_flags & 1
        self.data = raw_data[offset:]

class UDP:
    def __init__(self, raw_data):
        self.src_port, self.dest_port, self.size = struct.unpack('! H H 2x H', raw_data[:8])
        self.data = raw_data[8:]

class ARP:
    def __init__(self, raw_data):
        self.srcMAC, self.srcIP, self.targetMAC, self.targetIP = struct.unpack('! 8x 4s 4s 4s 4s', raw_data[:24])
        self.targetMAC = get_mac_addr(self.targetMAC)
        self.srcMAC = get_mac_addr(self.srcMAC)
        self.srcIP = socket.inet_ntoa(self.srcIP)
        self.targetIP = socket.inet_ntoa(self.targetIP)
        self.data = raw_data[24:]


# Returns MAC as string from bytes (ie AA:BB:CC:DD:EE:FF)
def get_mac_addr(mac_raw):
    byte_str = map('{:02x}'.format, mac_raw)
    mac_addr = ':'.join(byte_str).upper()
    return mac_addr