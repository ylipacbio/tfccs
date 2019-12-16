  $ DATADIR=$TESTDIR/../data/fextract2numpy
  $ PYBIN=$TESTDIR/../../tfccs/
  $ IN=$DATADIR/input.fextract.csv
  $ OUT_TRAIN=$CRAMTMP/output.train.npz

Test1: fextract2numpy exists
  $ fextract2numpy --help  1>&2 >/dev/null && echo $?
  0

Test2: run a small data
  $ fextract2numpy ${IN} ${CRAMTMP}/output --num-train-rows 3 1>&2 >/dev/null && echo $?
  0
  $ ls ${OUT_TRAIN} > /dev/null && echo $?
  0

Test3: emit stat-json
  $ DATADIR=$TESTDIR/../data/
  $ STAT_JSON=${DATADIR}/fextract.stat.json
  $ OUT_PREFIX=$CRAMTMP/test3_out
  $ fextract2numpy ${IN} ${OUT_PREFIX} --stat-json ${STAT_JSON} --num-train-rows 5 1>&2 >/dev/null && echo $?
  0

  $ ls ${OUT_PREFIX}.train.npz > /dev/null && echo $?
  0

  $ python -c "from tfccs.utils import load_fextract_npz; print(load_fextract_npz(\"${OUT_PREFIX}.train.npz\")[0][0])"
  [ 0.9853782   0.8212768  -0.7974602  -0.38539958 -0.54102606 -0.3833715
    0.44156098  0.67862433  1.2597709   0.9615064   1.292812   -0.5629895
    0.82436967  0.30031347 -0.31018588 -0.40606946 -0.39460257 -0.62491006
    0.0131169  -0.4351528  -0.35767877 -0.0051131  -0.38738513 -0.05900731
   -0.3234753  -0.38138753 -0.6246524  -0.39258978 -0.04281939 -0.31842884
   -0.37003654 -0.6249917  -0.05406152 -0.04836276 -0.36340305 -0.35072488
   -0.00528829 -0.06045663 -0.0356654  -0.36382577 -0.33770943 -0.00452501
   -0.7088231  -0.04577243 -0.4182008   1.2346268  -0.6245536  -0.4135914
   -0.04611078 -0.4099661  -0.482276   -0.62499106 -0.07425844 -0.03672864
   -0.36469027 -0.42603302 -0.00544887 -0.07099032  0.27443373 -0.36391178
   -0.43525255 -0.00417574 -0.6184471  -0.04576914  0.78694916 -0.54951936
   -0.6234667  -0.8162449  -0.05041147  2.0854146  -0.55101925 -0.6248201
   -0.07674053  0.27725208 -0.3718609  -0.49132738 -0.00563821 -0.07519425
   -0.07823464 -0.3724044  -0.4929441  -0.00375705 -0.42114714 -0.05030246
   -0.57040876 -0.61059475 -0.6231498  -0.4230069  -0.05826067 -0.57071745
   -0.6087259  -0.62521386 -0.07583258 -0.06316138 -0.37258163 -0.545369
   -0.00517165 -0.0773188  -0.0803782  -0.37342906 -0.54531455 -0.00312263
   -0.37701425  1.5980093   1.5972086   0.8775328   1.1618339  -0.48362666
   -0.45327428 -0.4839034   1.0808077   0.94427305 -0.41641396 -0.47403482
   -0.43089172  0.          0.          1.          0.        ]

  $ head -2 ${OUT_PREFIX}.train.fextract.csv
  Movie,HoleNumber,CCSBase,CCSPos,CCSLength,ArrowQv,CCSToGenomeStrand,CCSToGenomeCigar,IsCCSHP,CCSHPLength,CCSBaseSNR,MsaCoverage_FWD,MsaCoverage_REV,BaseCoverage_FWD,BaseCoverage_REV,SubreadHP0_FWD,SubreadHP0_REV,MeanSubreadHP0_FWD,MeanSubreadHP0_REV,MeanSubreadHP0,StdevSubreadHP0_FWD,StdevSubreadHP0_REV,StdevSubreadHP0,SeqMatch0_FWD,Insertion0_FWD,Deletion0_FWD,Substitution0_FWD,Unmapped0_FWD,SeqMatch0_REV,Insertion0_REV,Deletion0_REV,Substitution0_REV,Unmapped0_REV,SeqMatch1_FWD_PREV,Insertion1_FWD_PREV,Deletion1_FWD_PREV,Substitution1_FWD_PREV,Unmapped1_FWD_PREV,SeqMatch1_FWD_NEXT,Insertion1_FWD_NEXT,Deletion1_FWD_NEXT,Substitution1_FWD_NEXT,Unmapped1_FWD_NEXT,SeqMatch1_REV_PREV,Insertion1_REV_PREV,Deletion1_REV_PREV,Substitution1_REV_PREV,Unmapped1_REV_PREV,SeqMatch1_REV_NEXT,Insertion1_REV_NEXT,Deletion1_REV_NEXT,Substitution1_REV_NEXT,Unmapped1_REV_NEXT,SeqMatch3_FWD_PREV,Insertion3_FWD_PREV,Deletion3_FWD_PREV,Substitution3_FWD_PREV,Unmapped3_FWD_PREV,SeqMatch3_FWD_NEXT,Insertion3_FWD_NEXT,Deletion3_FWD_NEXT,Substitution3_FWD_NEXT,Unmapped3_FWD_NEXT,SeqMatch3_REV_PREV,Insertion3_REV_PREV,Deletion3_REV_PREV,Substitution3_REV_PREV,Unmapped3_REV_PREV,SeqMatch3_REV_NEXT,Insertion3_REV_NEXT,Deletion3_REV_NEXT,Substitution3_REV_NEXT,Unmapped3_REV_NEXT,SeqMatch6_FWD_PREV,Insertion6_FWD_PREV,Deletion6_FWD_PREV,Substitution6_FWD_PREV,Unmapped6_FWD_PREV,SeqMatch6_FWD_NEXT,Insertion6_FWD_NEXT,Deletion6_FWD_NEXT,Substitution6_FWD_NEXT,Unmapped6_FWD_NEXT,SeqMatch6_REV_PREV,Insertion6_REV_PREV,Deletion6_REV_PREV,Substitution6_REV_PREV,Unmapped6_REV_PREV,SeqMatch6_REV_NEXT,Insertion6_REV_NEXT,Deletion6_REV_NEXT,Substitution6_REV_NEXT,Unmapped6_REV_NEXT,SeqMatch10_FWD_PREV,Insertion10_FWD_PREV,Deletion10_FWD_PREV,Substitution10_FWD_PREV,Unmapped10_FWD_PREV,SeqMatch10_FWD_NEXT,Insertion10_FWD_NEXT,Deletion10_FWD_NEXT,Substitution10_FWD_NEXT,Unmapped10_FWD_NEXT,SeqMatch10_REV_PREV,Insertion10_REV_PREV,Deletion10_REV_PREV,Substitution10_REV_PREV,Unmapped10_REV_PREV,SeqMatch10_REV_NEXT,Insertion10_REV_NEXT,Deletion10_REV_NEXT,Substitution10_REV_NEXT,Unmapped10_REV_NEXT,PosInCcsHP,SNR_A,SNR_C,SNR_G,SNR_T,Subread_A_Count_FWD,Subread_A_Count_REV,Subread_C_Count_FWD,Subread_C_Count_REV,Subread_G_Count_FWD,Subread_G_Count_REV,Subread_T_Count_FWD,Subread_T_Count_REV,PrevCcsToGenomeCigar,NextCcsToGenomeCigar,CcsToGenomePrevDeletions
  m64002_190606_200346,458830,G,9483,19930,30,F,=,1,3,4.66223,2,3,2,2,2,2,3,2.33333,2.6,0,0.942809,0.8,2,0,0,0,0,2,0,0,0,1,2,0,0,0,0,2,0,0,0,0,2,0,0,0,1,2,0,0,0,1,3,0,0,1,0,4,0,0,0,0,4,0,0,0,2,4,2,0,0,2,5,0,1,0,0,4,0,2,0,0,6,3,0,0,3,6,0,0,0,3,8,0,0,0,0,8,0,0,0,0,8,0,0,0,4,8,0,0,0,4,0,13.7927,20.26,4.66223,8.44885,0,0,0,2,2,0,0,0,=,=,0

  $ cat ${OUT_PREFIX}.features.order.json
  {
      "OrderedFeatures": [
          "IsCCSHP",
          "CCSHPLength",
          "MsaCoverage_FWD",
          "MsaCoverage_REV",
          "BaseCoverage_FWD",
          "BaseCoverage_REV",
          "SubreadHP0_FWD",
          "SubreadHP0_REV",
          "MeanSubreadHP0_FWD",
          "MeanSubreadHP0_REV",
          "MeanSubreadHP0",
          "StdevSubreadHP0_FWD",
          "StdevSubreadHP0_REV",
          "StdevSubreadHP0",
          "SeqMatch0_FWD",
          "Deletion0_FWD",
          "Substitution0_FWD",
          "Unmapped0_FWD",
          "SeqMatch0_REV",
          "Deletion0_REV",
          "Substitution0_REV",
          "Unmapped0_REV",
          "SeqMatch1_FWD_PREV",
          "Insertion1_FWD_PREV",
          "Deletion1_FWD_PREV",
          "Substitution1_FWD_PREV",
          "Unmapped1_FWD_PREV",
          "SeqMatch1_FWD_NEXT",
          "Insertion1_FWD_NEXT",
          "Deletion1_FWD_NEXT",
          "Substitution1_FWD_NEXT",
          "Unmapped1_FWD_NEXT",
          "SeqMatch1_REV_PREV",
          "Insertion1_REV_PREV",
          "Deletion1_REV_PREV",
          "Substitution1_REV_PREV",
          "Unmapped1_REV_PREV",
          "SeqMatch1_REV_NEXT",
          "Insertion1_REV_NEXT",
          "Deletion1_REV_NEXT",
          "Substitution1_REV_NEXT",
          "Unmapped1_REV_NEXT",
          "SeqMatch3_FWD_PREV",
          "Insertion3_FWD_PREV",
          "Deletion3_FWD_PREV",
          "Substitution3_FWD_PREV",
          "Unmapped3_FWD_PREV",
          "SeqMatch3_FWD_NEXT",
          "Insertion3_FWD_NEXT",
          "Deletion3_FWD_NEXT",
          "Substitution3_FWD_NEXT",
          "Unmapped3_FWD_NEXT",
          "SeqMatch3_REV_PREV",
          "Insertion3_REV_PREV",
          "Deletion3_REV_PREV",
          "Substitution3_REV_PREV",
          "Unmapped3_REV_PREV",
          "SeqMatch3_REV_NEXT",
          "Insertion3_REV_NEXT",
          "Deletion3_REV_NEXT",
          "Substitution3_REV_NEXT",
          "Unmapped3_REV_NEXT",
          "SeqMatch6_FWD_PREV",
          "Insertion6_FWD_PREV",
          "Deletion6_FWD_PREV",
          "Substitution6_FWD_PREV",
          "Unmapped6_FWD_PREV",
          "SeqMatch6_FWD_NEXT",
          "Insertion6_FWD_NEXT",
          "Deletion6_FWD_NEXT",
          "Substitution6_FWD_NEXT",
          "Unmapped6_FWD_NEXT",
          "SeqMatch6_REV_PREV",
          "Insertion6_REV_PREV",
          "Deletion6_REV_PREV",
          "Substitution6_REV_PREV",
          "Unmapped6_REV_PREV",
          "SeqMatch6_REV_NEXT",
          "Insertion6_REV_NEXT",
          "Deletion6_REV_NEXT",
          "Substitution6_REV_NEXT",
          "Unmapped6_REV_NEXT",
          "SeqMatch10_FWD_PREV",
          "Insertion10_FWD_PREV",
          "Deletion10_FWD_PREV",
          "Substitution10_FWD_PREV",
          "Unmapped10_FWD_PREV",
          "SeqMatch10_FWD_NEXT",
          "Insertion10_FWD_NEXT",
          "Deletion10_FWD_NEXT",
          "Substitution10_FWD_NEXT",
          "Unmapped10_FWD_NEXT",
          "SeqMatch10_REV_PREV",
          "Insertion10_REV_PREV",
          "Deletion10_REV_PREV",
          "Substitution10_REV_PREV",
          "Unmapped10_REV_PREV",
          "SeqMatch10_REV_NEXT",
          "Insertion10_REV_NEXT",
          "Deletion10_REV_NEXT",
          "Substitution10_REV_NEXT",
          "Unmapped10_REV_NEXT",
          "PosInCcsHP",
          "SNR_A",
          "SNR_C",
          "SNR_G",
          "SNR_T",
          "Subread_A_Count_FWD",
          "Subread_A_Count_REV",
          "Subread_C_Count_FWD",
          "Subread_C_Count_REV",
          "Subread_G_Count_FWD",
          "Subread_G_Count_REV",
          "Subread_T_Count_FWD",
          "Subread_T_Count_REV",
          "CCSBaseA",
          "CCSBaseC",
          "CCSBaseG",
          "CCSBaseT"
      ]
  } (no-eol)
