import numpy as np
import random
import sys
import zipfile
import os
import multiprocessing as mp
import time

################################
def timeit( func, *args, **kwargs ):
    ts = time.time()
    tmp = func(*args, **kwargs)
    te = time.time()
    print("timeit", (te-ts))
    return(tmp)

#def mult(x,y):
#    return(x*y)
#timeit(mult, 5,3)

################################
def loadonefile( filename, myzfname=None ):
    print("loadonefile",  filename, "from", myzfname)
    prefix = ""
    if myzfname is not None: 
        zf = zipfile.ZipFile(myzfname, mode='r') # parallel: can't pass zf without corruption so open here!
        print("loadonefile extract")
        prefix=""
        timeit( zf.extract, filename) #, path=prefix )
        zf.close()
    print("loadonefile np.load")
    tmp = timeit( np.load, prefix+filename )
    #print("loadonefile",  filename, "tmp['windowinput'].shape", tmp['windowinput'].shape)
    if myzfname is not None: os.remove(prefix+filename)
    return(tmp)

################################
def myconcatenate( arrays, **kwargs ):
    """"concatenate / append two numpy arrays
        This is slow and takes linear time: 
        inputdat = timeit(np.concatenate, (inputdat, tmprs), axis=0)
    """
    ar1shape = arrays[0].shape
    ar2shape = arrays[1].shape
    if not len(ar1shape)==len(ar2shape):
        print("ERROR can't concat arrays of different dimensions")
        sys.exit(1)
    arnewshape = list(ar1shape)
    arnewshape[0] = ar1shape[0]+ar2shape[0]
    arnew = np.empty(arnewshape)
    arnew[:ar1shape[0]] = arrays[0]
    arnew[ar1shape[0]:arnewshape[0]] = arrays[1]
    return(arnew)

################################
class data:
    ################################
    def __init__(self, batch_size, datafile, shortcut=False):

        self.batch_size = batch_size
        self.datafile = datafile

        # TODO: npz load here to see if it cuts the time!
        mydat = self.loaddata( datafile, isZip=False, shortcut=shortcut)
        self.inputdat = mydat["inputdat"]
        self.outputdat = mydat["outputdat"]

        self.create_batches()
        self.reset_batch_pointer()

    ################################
    def loaddata( self, datafile, isZip=False, shortcut=False ):
        # shortcut only loads first of first

        ####
        def loadfilenames( filenames, zfname=None ):
            # first
            for ii in [0]:
                df = filenames[ii]
                tmp = loadonefile(df, myzfname=zfname)
                inputdat = tmp['windowinput']
                outputdat = tmp['windowoutput']
            # rest

            if not shortcut:
                if True:
                    for ii in range(1,len(filenames)):
                        df = filenames[ii]
                        tmp = loadonefile(df, myzfname=zfname)
                        tmprs = tmp['windowinput']
                        print("loadfilenames concat1")
                        inputdat = timeit(np.concatenate, (inputdat, tmprs), axis=0)
                        tmprs = tmp['windowoutput']
                        print("loadfilenames concat2")
                        outputdat = timeit(np.concatenate, (outputdat, tmprs), axis=0)

            #print("loadfilenames final inputdat.shape", inputdat.shape)
            #print("loadfilenames final outputdat.shape", outputdat.shape)

            return( { "inputdat": inputdat, "outputdat": outputdat} )
        ####

        #### filenames can be a list of zip files with ~200 numpy arrays.
        #### load all numpy arrays in each zip file
        filenames=open(datafile).read().splitlines()
        if not isZip:
            mydat = loadfilenames( filenames )
            return( mydat ) # { "inputdat": mydat["inputdat"], "outputdat": mydat["outputdat"]} )
        else:
            first=True
            for ff in filenames:
                print("loaddata zipfile",ff)
                zf = zipfile.ZipFile(ff, mode='r')
                filenames = zf.namelist()
                zf.close()
                mydat = loadfilenames(filenames, ff)
                if first:
                    inputdat = mydat["inputdat"]
                    outputdat = mydat["outputdat"]
                    first=False
                    if shortcut: break
                else:
                    inputdat = np.concatenate( (inputdat, mydat["inputdat"]),axis=0)
                    outputdat = np.concatenate( (outputdat, mydat["outputdat"]),axis=0)
            print("loaddata zip final inputdat.shape", inputdat.shape)
            print("loaddata zip final outputdat.shape", outputdat.shape)
            return( { "inputdat": inputdat, "outputdat": outputdat} )

    ################################
    # seperate the whole data into different batches.
    def create_batches(self):
        if (self.inputdat.shape[0] % self.batch_size) == 0:
            extra = 0
        else:
            extra = 1
        self.num_batches = int(self.inputdat.shape[0]/self.batch_size)+extra

    def reset_batch_pointer(self):
        self.pointer = 0

    def next_batch(self):
        mystart = self.pointer * self.batch_size
        myend = (self.pointer+1) * self.batch_size
        inputs = self.inputdat[mystart:myend,]
        outputs = self.outputdat[mystart:myend,]
        self.pointer += 1
        return( inputs, outputs )

def printit( x ):
    print("----")
    print(x.shape)
    for bb in range(x.shape[0]):
      feats = []    
      for ff in x[bb]:
        feats.append("%s" % str(ff))
      print("%d: %s" % (bb," ".join(feats)))

################################

if __name__ == "__main__":

    ### load the data from sys.argv[1]
    t0=time.time()
    data_loader = data( 128, sys.argv[1])
    t1=time.time()
    print("time data_loader",str(t1-t0))

    ### save the data as npz in sys.argv[2]
    t0=time.time()
    np.savez_compressed(sys.argv[2], windowinput=data_loader.inputdat, windowoutput=data_loader.outputdat)
    t1=time.time()
    print("time savez_compressed",str(t1-t0))

