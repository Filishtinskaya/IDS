import math
import numpy as np


class incStat:
    def __init__(self, Lambda, ID, init_time=0, isTypeDiff=False):  # timestamp is creation time
        self.ID = ID
        self.CF1 = 0  # linear sum
        self.CF2 = 0  # sum of squares
        self.w = 1e-20  # weight
        self.isTypeDiff = isTypeDiff
        self.Lambda = Lambda  # Decay Factor
        self.lastTimestamp = init_time
        self.cur_mean = np.nan
        self.cur_var = np.nan
        self.cur_std = np.nan
        self.covs = [] # a list of incStat_covs (references) with relate to this incStat

    def insert(self, v, t=0):  # v is a scalar, t is v's arrival the timestamp
        if self.isTypeDiff:
            if t - self.lastTimestamp > 0:
                v = t - self.lastTimestamp
            else:
                v = 0
        self.processDecay(t)

        # update with v
        self.CF1 += v
        self.CF2 += math.pow(v, 2)
        self.w += 1
        self.cur_mean = np.nan  # force recalculation if called
        self.cur_var = np.nan
        self.cur_std = np.nan

        # update covs (if any)
        for c in self.covs:
            cov = c
            cov.update_cov(self.ID, v, t)

    def processDecay(self, timestamp):
        factor = 1
        # check for decay
        timeDiff = timestamp - self.lastTimestamp
        if timeDiff > 0:
            factor = math.pow(2, (-self.Lambda * timeDiff))
            self.CF1 = self.CF1 * factor
            self.CF2 = self.CF2 * factor
            self.w = self.w * factor
            self.lastTimestamp = timestamp
        return factor

    def weight(self):
        return self.w

    def mean(self):
        if math.isnan(self.cur_mean):  # calculate it only once when necessary
            self.cur_mean = self.CF1 / self.w
        return self.cur_mean

    def var(self):
        if math.isnan(self.cur_var):  # calculate it only once when necessary
            self.cur_var = abs(self.CF2 / self.w - math.pow(self.mean(), 2))
        return self.cur_var

    def std(self):
        if math.isnan(self.cur_std):  # calculate it only once when necessary
            self.cur_std = math.sqrt(self.var())
        return self.cur_std

    def cov(self,ID2):
        for cov in self.covs:
            if cov.isRelated(ID2):
                return cov.cov()
        return [np.nan]

    def pcc(self,ID2):
        for cov in self.covs:
            if cov.isRelated(ID2):
                return cov.pcc()
        return [np.nan]

    def cov_pcc(self,ID2):
        for c in self.covs:
            cov = c
            if cov.isRelated(ID2):
                return cov.get_stats1()
        return [np.nan]*2

    def radius(self, other_incStats):  # the radius of a set of incStats
        A = self.var()
        for incS in other_incStats:
            incSc = incS
            A += incSc.var()
        return math.sqrt(A)

    def magnitude(self, other_incStats):  # the magnitude of a set of incStats
        A = math.pow(self.mean(), 2)
        for incS in other_incStats:
            incSc = incS
            A += math.pow(incSc.mean(), 2)
        return math.sqrt(A)

    #calculates and pulls all stats on this stream
    def allstats_1D(self):
        self.cur_mean = self.CF1 / self.w
        self.cur_var = abs(self.CF2 / self.w - math.pow(self.cur_mean, 2))
        return [self.w, self.cur_mean, self.cur_var]

    #calculates and pulls all stats on this stream, and stats shared with the indicated stream
    def allstats_2D(self, ID2):
        stats1D = self.allstats_1D()
        # Find cov component
        stats2D = [np.nan] * 4
        for c in self.covs:
            cov = c
            if cov.isRelated(ID2):
                stats2D = cov.get_stats2()
                break
        return stats1D + stats2D

    def getHeaders_1D(self, suffix=True):
        if self.ID is None:
            s0=""
        else:
            s0 = "_0"
        if suffix:
            s0 = "_"+self.ID
        headers = ["weight"+s0, "mean"+s0, "std"+s0]
        return headers

    def getHeaders_2D(self, ID2, suffix=True):
        hdrs1D = self.getHeaders_1D(suffix)
        if self.ID is None:
            s0=""
            s1=""
        else:
            s0 = "_0"
            s1 = "_1"
        if suffix:
            s0 = "_"+self.ID
            s1 = "_" + ID2
        hdrs2D = ["radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1, "covariance_" + s0 + "_" + s1,
                   "pcc_" + s0 + "_" + s1]
        return hdrs1D+hdrs2D

    # def toJSON(self):
    #     j = {}
    #     j['CF1'] = self.CF1
    #     j['CF2'] = self.CF2
    #     j['w'] = self.w
    #     j['isTypeDiff'] = self.isTypeDiff
    #     j['Lambda'] = self.Lambda
    #     j['lastTimestamp'] = self.lastTimestamp
    #     return json.dumps(j)
    #
    # def loadFromJSON(self,JSONstring):
    #     j = json.loads(JSONstring)
    #     self.CF1 = j['CF1']
    #     self.CF2 = j['CF2']
    #     self.w = j['w']
    #     self.isTypeDiff = j['isTypeDiff']
    #     self.Lambda = j['Lambda']
    #     self.lastTimestamp = j['lastTimestamp']

#like incStat, but maintains stats between two streams
#TODO: make it possble to call incstat magnitude and raduis withour list of incstsats (just single incstat objects) for cov.getstats2 typcast call
class incStat_cov:
    def __init__(self, incS1, incS2, init_time = 0):
        # store references tot he streams' incStats
        self.incS1 = incS1
        self.incS2 = incS2

        # init extrapolators
        self.ex1 = extrapolator()
        self.ex2 = extrapolator()

        # init sum product residuals
        self.CF3 = 0 # sum of residule products (A-uA)(B-uB)
        self.w3 = 1e-20
        self.lastTimestamp_cf3 = init_time

    #other_incS_decay is the decay factor of the other incstat
    # ID: the stream ID which produced (v,t)
    def update_cov(self, ID, v, t):  # it is assumes that incStat "ID" has ALREADY been updated with (t,v) [this si performed automatically in method incStat.insert()]
        # find incStat
        if ID == self.incS1.ID:
            inc = 0
        else:
            inc = 1

        # Decay residules
        self.processDecay(t)

        # Update extrapolator for current stream AND
        # Extrapolate other stream AND
        # Compute and update residule
        if inc == 0:
            self.ex1.insert(t,v)
            v_other = self.ex2.predict(t)
            self.CF3 += (v - self.incS1.mean()) * (v_other - self.incS2.mean())
        else:
            self.ex2.insert(t,v)
            v_other = self.ex1.predict(t)
            self.CF3 += (v - self.incS2.mean()) * (v_other - self.incS1.mean())
        self.w3 += 1

    def processDecay(self, t):
        factor = 1
        # check for decay cf3
        timeDiffs_cf3 = t - self.lastTimestamp_cf3
        if timeDiffs_cf3 > 0:
            factor = math.pow(2, (-(self.incS1.Lambda) * timeDiffs_cf3))
            self.CF3 *= factor
            self.w3 *= factor
            self.lastTimestamp_cf3 = t
        return factor

    #todo: add W3 for cf3

    #covariance approximation
    def cov(self):
        return self.CF3 / self.w3

    # Pearson corl. coef
    def pcc(self):
        ss = self.incS1.std() * self.incS2.std()
        if ss != 0:
            return self.cov() / ss
        else:
            return 0

    def isRelated(self, ID):
        if self.incS1.ID == ID or self.incS2.ID == ID:
            return True
        else:
            return False

    # calculates and pulls all correlative stats
    def get_stats1(self):
        return [self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND 2D stats from both streams (incStat)
    def get_stats2(self):
        return [self.incS1.radius([self.incS2]),self.incS1.magnitude([self.incS2]),self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND 2D stats AND the regular stats from both streams (incStat)
    def get_stats3(self):
        return [self.incS1.w,self.incS1.mean(),self.incS1.std(),self.incS2.w,self.incS2.mean(),self.incS2.std(),self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND the regular stats from both incStats AND 2D stats
    def get_stats4(self):
        return [self.incS1.w,self.incS1.mean(),self.incS1.std(),self.incS2.w,self.incS2.mean(),self.incS2.std(), self.incS1.radius([self.incS2]),self.incS1.magnitude([self.incS2]),self.cov(), self.pcc()]

    def getHeaders(self,ver,suffix=True): #ver = {1,2,3,4}
        headers = []
        s0 = "0"
        s1 = "1"
        if suffix:
            s0 = self.incS1.ID
            s1 = self.incS2.ID

        if ver == 1:
            headers = ["covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 2:
            headers = ["radius_"+s0+"_"+s1, "magnitude_"+s0+"_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 3:
            headers = ["weight_"+s0, "mean_"+s0, "std_"+s0,"weight_"+s1, "mean_"+s1, "std_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 4:
            headers = ["weight_" + s0, "mean_" + s0, "std_" + s0, "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 5:
            headers = ["weight_"+s0, "mean_"+s0, "std_"+s0,"weight_"+s1, "mean_"+s1, "std_"+s1, "radius_"+s0+"_"+s1, "magnitude_"+s0+"_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        return headers


class incStatDB:
    # default_lambda: use this as the lambda for all streams. If not specified, then you must supply a Lambda with every query.
    def __init__(self, limit=np.Inf, default_lambda=np.nan):
        self.HT = dict()
        self.limit = limit
        self.df_lambda = default_lambda

    def get_lambda(self, Lambda):
        if not np.isnan(self.df_lambda):
            Lambda = self.df_lambda
        return Lambda

    # Registers a new stream. init_time: init lastTimestamp of the incStat
    def register(self,ID, Lambda=1, init_time=0,isTypeDiff=False):
        #Default Lambda?
        Lambda = self.get_lambda(Lambda)

        #Retrieve incStat
        key = ID+"_"+str(Lambda)

        incS = self.HT.get(key)
        if incS is None: #does not already exist
            if len(self.HT) + 1 > self.limit:
                raise LookupError(
                    'Adding Entry:\n' + key + '\nwould exceed incStatHT 1D limit of ' + str(
                        self.limit) + '.\nObservation Rejected.')
            incS = incStat(Lambda, ID, init_time, isTypeDiff)
            self.HT[key] = incS #add new entry
        return incS

    # Registers covariance tracking for two streams, registers missing streams
    def register_cov(self,ID1, ID2, Lambda=1, init_time=0, isTypeDiff=False):
        #Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Lookup both streams
        incS1 = self.register(ID1,Lambda,init_time,isTypeDiff)
        incS2 = self.register(ID2,Lambda,init_time,isTypeDiff)

        #check for pre-exiting link
        for cov in incS1.covs:
            if cov.isRelated(ID2):
                return cov #there is a pre-exiting link

        # Link incStats
        inc_cov = incStat_cov(incS1,incS2,init_time)
        incS1.covs.append(inc_cov)
        incS2.covs.append(inc_cov)
        return inc_cov

    # updates/registers stream
    def update(self,ID, t, v, Lambda=1,isTypeDiff=False):
        incS = self.register(ID,Lambda,t,isTypeDiff)
        incS.insert(v,t)
        return incS

    # Pulls current stats from the given ID
    def get_1D_Stats(self,ID, Lambda=1): #weight, mean, std
        #Default Lambda?
        Lambda = self.get_lambda(Lambda)

        #Get incStat
        incS = self.HT.get(ID+"_"+str(Lambda))
        if incS is None:  # does not already exist
            return [np.na]*3
        else:
            return incS.allstats_1D()

    # Pulls current correlational stats from the given IDs
    def get_2D_Stats(self, ID1, ID2, Lambda=1): #cov, pcc
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStat
        incS = self.HT.get(ID1 + "_" + str(Lambda))
        if incS is None:  # does not exist
            return [np.na]*2

        # find relevant cov entry
        return incS.cov_pcc(ID2)

    # Pulls all correlational stats registered with the given ID
    # returns tuple [0]: stats-covs&pccs, [2]: IDs
    def get_all_2D_Stats(self, ID, Lambda=1):  # cov, pcc
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStat
        incS1 = self.HT.get(ID + "_" + str(Lambda))
        if incS1 is None:  # does not exist
            return ([],[])

        # find relevant cov entry
        stats = []
        IDs = []
        for cov in incS1.covs:
            stats.append(cov.get_stats1())
            IDs.append([cov.incS1.ID,cov.incS2.ID])
        return stats,IDs

    # Pulls current multidimensional stats from the given IDs
    def get_nD_Stats(self,IDs, Lambda=1): #radius, magnitude (IDs is a list)
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStats
        incStats = []
        for ID in IDs:
            incS = self.HT.get(ID + "_" + str(Lambda))
            if incS is not None:  #exists
                incStats.append(incS)

        # Compute stats
        rad = 0 #radius
        mag = 0 #magnitude
        for incS in incStats:
            rad += incS.var()
            mag += incS.mean()**2

        return [np.sqrt(rad),np.sqrt(mag)]

    # Updates and then pulls current 1D stats from the given ID. Automatically registers previously unknown stream IDs
    def update_get_1D_Stats(self, ID, t, v, Lambda=1, isTypeDiff=False):  # weight, mean, std

        incS = self.update(ID,t,v,Lambda,isTypeDiff)
        return incS.allstats_1D()


    # Updates and then pulls current correlative stats between the given IDs. Automatically registers previously unknown stream IDs, and cov tracking
    #Note: AfterImage does not currently support Diff Type streams for correlational statistics.
    def update_get_2D_Stats(self, ID1, ID2, t1, v1, Lambda=1, level=1):  #level=  1:cov,pcc  2:radius,magnitude,cov,pcc
        #retrieve/add cov tracker
        inc_cov = self.register_cov(ID1, ID2, Lambda,  t1)
        # Update cov tracker
        inc_cov.update_cov(ID1,v1,t1)
        if level == 1:
            return inc_cov.get_stats1()
        else:
            return inc_cov.get_stats2()

    # Updates and then pulls current 1D and 2D stats from the given IDs. Automatically registers previously unknown stream IDs
    def update_get_1D2D_Stats(self, ID1, ID2, t1, v1, Lambda=1):  # weight, mean, std
        return self.update_get_1D_Stats(ID1,t1,v1,Lambda) + self.update_get_2D_Stats(ID1,ID2,t1,v1,Lambda,level=2)

    def getHeaders_1D(self,Lambda=1,ID=''):
        # Default Lambda?
        L = Lambda
        L = self.get_lambda(L)
        hdrs = incStat(L,ID).getHeaders_1D(suffix=False)
        return [str(L)+"_"+s for s in hdrs]

    def getHeaders_2D(self,Lambda=1,IDs=None, ver=1): #IDs is a 2-element list or tuple
        # Default Lambda?
        L = Lambda
        L = self.get_lambda(L)
        if IDs is None:
            IDs = ['0','1']
        hdrs = incStat_cov(incStat(L,IDs[0]),incStat(L,IDs[0]),L).getHeaders(ver,suffix=False)
        return [str(Lambda)+"_"+s for s in hdrs]

    def getHeaders_1D2D(self,Lambda=1,IDs=None, ver=1):
        # Default Lambda?
        L = Lambda
        L = self.get_lambda(L)
        if IDs is None:
            IDs = ['0','1']
        hdrs1D = self.getHeaders_1D(L,IDs[0])
        hdrs2D = self.getHeaders_2D(L,IDs, ver)
        return hdrs1D + hdrs2D

    def getHeaders_nD(self,Lambda=1,IDs=[]): #IDs is a n-element list or tuple
        # Default Lambda?
        L = Lambda
        ID = ":"
        for s in IDs:
            ID += "_"+s
        L = self.get_lambda(L)
        hdrs = ["radius"+ID, "magnitude"+ID]
        return [str(L)+"_"+s for s in hdrs]


    #cleans out records that have a weight less than the cutoff.
    #returns number or removed records.
    def cleanOutOldRecords(self, cutoffWeight, curTime):
        n = 0
        dump = sorted(self.HT.items(), key=lambda tup: tup[1][0].getMaxW(curTime))
        for entry in dump:
            entry[1][0].processDecay(curTime)
            W = entry[1][0].w
            if W <= cutoffWeight:
                key = entry[0]
                del entry[1][0]
                del self.HT[key]
                n=n+1
            elif W > cutoffWeight:
                break
        return n




class incHist:
    #ubIsAnom means that the HBOS score for vals that fall past the upped bound are Inf (not 0)
    def __init__(self,nbins,Lambda=0,ubIsAnom=True,lbIsAnom=True,lbound=-10,ubound=10,scaleGrace=None):
        self.scaleGrace = scaleGrace #the numbe rof instances to observe until a range it determeined
        if scaleGrace is not None:
            self.lbound = np.Inf
            self.ubound = -np.Inf
            self.binSize = None
            self.isScaling = True
        else:
            self.lbound = lbound
            self.ubound = ubound
            self.binSize = (ubound - lbound)/nbins
            self.isScaling = False
        self.nbins = nbins
        self.ubIsAnom = ubIsAnom
        self.lbIsAnom = lbIsAnom
        self.n = 0

        self.Lambda = Lambda
        self.W = np.zeros(nbins)
        self.lT = np.zeros(nbins) #last timestamp of each respective bin
        self.tallestBin = 0 #indx to the bin that currently has the largest freq weight (assumed...)

    #assumes even bin width starting from lbound until ubound. beyond bounds are assigned to the closest bin
    def getBinIndx(self,val,win=0):
        indx = int(np.floor((val - self.lbound)/self.binSize))
        if win == 0:
            if indx < 0:
                return -np.Inf
            if indx > (self.nbins - 1):
                return np.Inf
            return indx
        else: #windowed Histogram
            if indx - win < 0: #does the left of the window stick out of bounds?
                if indx + win >= 0: #if yes, then is there some overlap with inbounds?
                    return range(0,indx+win+1) #return the inbounds range
                else: #then the entire window is our of bounds to the left
                    return -np.Inf
            if indx + win > self.nbins - 1: #does the right of the window stick out of bounds?
                if indx - win < self.nbins: #if yes, then is there some overlap with inbounds?
                    return range(indx - win,self.nbins) #return the inbounds range
                else: #then the entire window is our of bounds to the right
                    return np.Inf
            return range(indx-win,indx+win+1)


    def processDecay(self, bin, timestamp):
        # check for decay
        timeDiff = timestamp - self.lT[bin]
        if np.isscalar(timeDiff):
            if timeDiff > 0:
                factor = math.pow(2, (-self.Lambda * timeDiff))
                self.W[bin] = self.W[bin] * factor
                self.lT[bin] = timestamp
        else: #array
            timeDiff[timeDiff<0]=0 #don't affect decay of out of order entries
            factor = np.power(2, (-self.Lambda * timeDiff))
            #b4 = self.W[bin]
            self.W[bin] = self.W[bin] * factor
            self.lT[bin] = timestamp

    def insert(self,val,timestamp,penalty=False):
        self.n = self.n + 1
        if self.isScaling:
            if self.n < self.scaleGrace:
                if self.lbound > val:
                    self.lbound = val
                if self.ubound < val:
                    self.ubound = val
            if self.n == self.scaleGrace:
                if self.ubound == self.lbound:
                    self.scaleGrace = self.scaleGrace + 1000
                else:
                    width = self.ubound - self.lbound
                    self.ubound = self.ubound + width
                    self.lbound = self.lbound - width
                    self.binSize = (self.ubound - self.lbound) / self.nbins
                    self.isScaling = False
        else:
            bin = self.getBinIndx(val)
            if not np.isinf(bin): #
                self.processDecay(bin, timestamp)
                if penalty:
                    tallestW = self.W[self.tallestBin]
                    scale = tallestW if tallestW > 0 else 1
                    fn = self.W[bin]/scale
                    inc = self.halfsigmoid(fn+0.005,-1.03)
                else:
                    inc = 1
                self.W[bin] = self.W[bin] + inc
                #track who has the tallest bin (for normilization)
                if self.W[bin] > self.W[self.tallestBin]:
                    self.tallestBin = bin

    def halfsigmoid(self,x,k):
        return (k*x)/(k-x+1)

    def score(self,val,timestamp=-1,win=0): #HBOS for one dimension
        if self.isScaling:
            return 0.0
        else:
            bin = self.getBinIndx(val,win=win)
            if np.isscalar(bin):
                if np.isinf(bin):
                    if self.ubIsAnom and bin > 0:
                        return np.Inf #it's an anomaly because it passes the upper bound
                    elif self.lbIsAnom and bin < 0:
                        return np.Inf  # it's an anomaly because it passes the lower bound
                    else:
                        return 0.0 #it fell outside a bound which is consedered not anomalous
            self.processDecay(bin,timestamp) #if timestamp = -1, no decay will be applied
            w = np.mean(self.W[bin])
            if w == 0:
                return np.Inf  # no stat history, anomaly!
            else:
                return np.log(self.W[self.tallestBin] / (w))  # log(  1/(  p/p_max  )    )


    def getFreq(self,val,timestamp=-1): #HBOS for one dimension
        bin = self.getBinIndx(val)
        self.processDecay(bin,timestamp) #if timestamp = -1, no decay will be applied
        if np.isinf(bin):
            return np.nan
        else:
            return self.W[bin]

    def getHist(self,timestamp=-1): #HBOS for one dimension
        H = np.zeros((len(self.W),1))
        for i in range(0,len(self.W)):
            self.processDecay(i,timestamp) #if timestamp = -1, no decay will be applied
            H[i] = self.W[i]
        H = H/np.sum(self.W)
        return H
    #
    # def loadFromJSON(self,jsonstring):
    #     return '' # !!!! very  important: all timestamps in self.lT should be updated so the decay won't wipe out the histogram:
    #             # self.lT = self.lT + curtime - max(self.lT)
    #             # this also applies to when the system.train setting is toggled to 'on'

class Queue:

    q = [0, 0, 0]

    def __init__(self):
        self.q[0] = self.q[1] = self.q[2] = 0
        self.indx = 0
        self.n = 0

    def insert(self, v):
        self.q[self.indx] = v
        self.indx = (self.indx + 1) % 3
        self.n += 1

    def unroll(self):

        res = [0, 0]
        if self.n == 2:
            res[0] = self.q[0]
            res[1] = self.q[1]
            return res
        if self.indx == 0:
            return self.q

        res3 = [0, 0, 0]
        if self.indx == 1:
            res3[0] = self.q[1]
            res3[1] = self.q[2]
            res3[2] = self.q[0]
            return res3
        else:
            res3[0] = self.q[2]
            res3[1] = self.q[0]
            res3[2] = self.q[1]
            return res3

    def get_last(self):
        return self.q[self.indx]

    def get_mean_diff(self):
        dif = 0
        if self.n == 2:
            dif=self.q[self.indx%3] - self.q[(self.indx-1)%3]
            return dif
        else:
            # for i in range(2):
            #     dif+=self.q[(self.indx+i+1)%3] - self.q[(self.indx+i)%3]
            dif= (self.q[self.indx%3] - self.q[(self.indx-1)%3]) + (self.q[(self.indx-1)%3] - self.q[(self.indx-2)%3])
            return dif/2

class extrapolator:

    def __init__(self):#,winsize=3):
        self.Qt = Queue() #deque([],winsize) #window of timestamps
        self.Qv = Queue() #deque([],winsize) #window of values

    def insert(self, t, v):
        self.Qt.insert(t)
        self.Qv.insert(v)

    def predict(self, t):
        if self.Qt.n < 2: #not enough points to extrapolate?
            if self.Qt.n == 1:
                return self.Qv.get_last()
            else:
                return 0
        if (t - self.Qt.get_last())/(self.Qt.get_mean_diff() + 1e-10) > 10: # is the next timestamp 10 time further than the average sample interval?
            return self.Qv.get_last() # prediction too far ahead (very likely that we will be way off)

        tm = self.Qt.unroll()
        vm = self.Qv.unroll()
        yp = self.interpolate(t,tm,vm)
        return yp


    def interpolate(self, tp, tm, ym):
        n = len(tm) - 1
        #cdef[:] lagrpoly = np.array([self.lagrange(tp, i, tm) for i in range(n + 1)])

        for i in range(n +1):
            """
            Evaluate the i-th Lagrange polynomial at x
            based on grid data xm
            """
            y = 1
            for j in range(n + 1):
                if i != j:
                    try:
                        y *= (tp - tm[j]) / (tm[i] - tm[j] + 1e-20)
                    except ZeroDivisionError:
                        y*= (tp - tm[j]) / (tm[i] - tm[j])

            ym[i]*=y

        return sum(ym)

