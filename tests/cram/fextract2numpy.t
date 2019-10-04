  $ DATADIR=$TESTDIR/../data/fextract2numpy
  $ PYBIN=$TESTDIR/../../tfccs/
  $ IN=$DATADIR/input.fextract.csv
  $ OUT_TRAIN=$CRAMTMP/output.train.npz
  $ OUT_TEST=$CRAMTMP/output.test.npz

  $ python $PYBIN/fextract2numpy.py --help  1>&2 >/dev/null && echo $?
  0

  $ python $PYBIN/fextract2numpy.py ${IN} ${CRAMTMP}/output --num-train-rows 50 1>&2 >/dev/null && echo $?
  0
  $ ls ${OUT_TRAIN} > /dev/null && echo $?
  0
  $ ls ${OUT_TEST} > /dev/null && echo $?
  0
