  $ DATADIR=$TESTDIR/../data/fextract2numpy
  $ PYBIN=$TESTDIR/../../tfccs/
  $ IN=$DATADIR/input.fextract.csv
  $ OUT_TRAIN=$CRAMTMP/output.train.npz
  $ OUT_TEST=$CRAMTMP/output.test.npz

  $ fextract2numpy --help  1>&2 >/dev/null && echo $?
  0

  $ fextract2numpy ${IN} ${CRAMTMP}/output --num-train-rows 3 1>&2 >/dev/null && echo $?
  0
  $ ls ${OUT_TRAIN} > /dev/null && echo $?
  0
  $ ls ${OUT_TEST} > /dev/null && echo $?
  0
