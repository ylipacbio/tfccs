#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as KK
from .model import Model
from . import data

################################
def test(args):

    data_loader = data.data( args.batch_size, sys.argv[2],
                             inputdatName=args.inputdatName,
                             outputdatName=args.outputdatName)

    #with tf.device("/gpu:2"):
    if True:

        # # check compatibility if training is continued from previously saved model
        # if args.init_from is not None:
        #     # check if all necessary files exist
        #     assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        #     ckpt = tf.train.latest_checkpoint(args.init_from)
        #     assert ckpt, "No checkpoint found"
        #     print("ckpt", ckpt, file=sys.stderr)
        # if not os.path.isdir(args.save_dir):
        #     os.makedirs(args.save_dir)

        model = Model(args)

        # make it appear as though there is only one gpu and use it
        os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES

        num=0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            # saver = tf.train.Saver(tf.global_variables())
            # # restore model
            # if args.init_from is not None:
            #     saver.restore(sess, ckpt)

            # restore model
            model.model = KK.models.load_model("my_model_FULL.h5")

            ################################
            numerr = 0
            total = 0
            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()
                
                predictions = model.model.predict(x)

                print("predictions[0].shape",predictions[0].shape)
                print("y.shape",y.shape)

                #np.save("test.0.predictions",predictions[0])
                #np.save("test.1.predictions",predictions[1])

                #### print out all predictions
                if False:
                    fp = open("test.1.preds.txt","w")
                    for ii in range(predictions[1].shape[0]):
                        print("0\t%d\t%d\t%s\t-1\t%s" % (ii,
                                                         0,
                                                         "\t".join([str(xx) for xx in predictions[1][ii,:]]),
                                                         "\t".join([str(xx) for xx in y[ii,:]])
                                                     ), file=fp)
                    fp.close()
                end = time.time()

                ################################
                # take max for error rate and consensus call
                if True:
                    idx2Base = ["A","C","G","T",""]
                    consensusSeq = []
                    trueSeq = []
                    for obj in range(y.shape[0]):
                        consensusSeq.append( [] )
                        trueSeq.append( [] )
                        for objelement in range(y.shape[1]):
                            truth = y[obj,objelement]
                            estimate = predictions[obj,objelement]
                            truemax = np.argmax(truth)
                            estmax = np.argmax(estimate)
                            consensusSeq[obj].append(idx2Base[estmax])
                            trueSeq[obj].append(idx2Base[truemax])
                            # # estsort = np.sort(-estimate,1)

                            if truemax!=estmax:
                                numerr+=1
                                print("err1 %d %d true est prob" % (obj, objelement), truemax,estmax,estimate[estmax])
                            total+=1
                    print("BATCH %d 1 %f = %d / %d" % (b,float(numerr)/total,numerr,total))

                    #### evaluate full seqs
                    fptrue = open("seqtrue.fa","a")
                    fpest = open("seqest.fa","a")
                    for obj in range(len(trueSeq)):
                        fptrue.write(">true-%d-%d\n" % (b,obj))
                        fptrue.write("".join(trueSeq[obj]))
                        fptrue.write("\n")

                        fpest.write(">est-%d-%d\n" % (b,obj))
                        fpest.write("".join(consensusSeq[obj]))
                        fpest.write("\n")
                    fptrue.close()
                    fpest.close()
                            
                #break # only look at 0th batch for time
            print("error rate 1 %f = %d / %d" % (float(numerr)/total,numerr,total))

if __name__ == '__main__':
    exec(open(sys.argv[1]).read())

    for aa in sys.argv:
        if "EXEC:" in aa:
            toexec = aa.replace("EXEC:","")
            print("toexec",toexec)
            exec(toexec)
            
    print("-------")
    print(help(args))
    print("-------")

    test(args)
