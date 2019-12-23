set -vex -o pipefail
# This documented how I preprocess lambda digest dataset and train
# a multinomial model regression to CCS to genome mapping.
movie=m64002_190608_021007
echo "Movie: $movie"
original_db_dir=/pbi/dept/consensus/ccsqv/data/Mule/lambda/bam/m64002_190608_021007/V2Bmark_SMS_Beta2_LambdaDigestEagI_SP2p1_DA011306_FCR_pkmid500_3260307/
fextract_dir=/pbi/dept/secondary/siv/yli/jira/tak-230-tf-sampling

function sampling_xyz() {
    chunk_id=$1
    x=$2
    y=$3
    z=$4
    k=$5

    echo "chunk_id: ${chunk_id}"
    echo "X:Y:Z=$x,$y,$z"
    echo "K=$k"
    out_dir=XYZK_${x}_${y}_${z}_${k}
    mkdir -p ${out_dir}

    sr2ccs=${original_db_dir}/sr2ccs/lambdaEagI_G_${chunk_id}_sr2ccs.bam
    ccs2genome=${original_db_dir}/ArrowQv/chunk-${chunk_id}.ccs2genome.bam

    # input_fextract_csv=${out_dir}/chunk-${chunk_id}.fextract.csv
    # output_stat_json=${out_dir}/chunk-${chunk_id}.stat.json

    echo """set -vex -o pipefail
$PI/ccsqv/build-release/src/fextract  \\
    ${sr2ccs} \\
    ${ccs2genome} \\
    ${out_dir}/chunk-${chunk_id}.fextract.csv \\
    -r $x:$y:$z \\
    -k $k \\
    --forward-only -j 16 \\
    --log-level INFO \\
    --log-file $out_dir/chunk-${chunk_id}.log

echo "Need to run on tfbuild: fextract2stat ${input_fextract_csv} ${output_stat_json}"

""" > $out_dir/chunk-${chunk_id}.fextract.sh
    $pitch7/bin/qu 16 "bash $out_dir/chunk-${chunk_id}.fextract.sh"
}

function gen_stat (){
    chunk_id=$1
    x=$2
    y=$3
    z=$4
    k=$5

    echo "chunk_id: ${chunk_id}"
    echo "X:Y:Z=$x,$y,$z"
    echo "K=$k"
    out_dir=${fextract_dir}/XYZK_${x}_${y}_${z}_${k}
    input_fextract_csv=${out_dir}/chunk-${chunk_id}.fextract.csv
    output_stat_json=${out_dir}/chunk-${chunk_id}.stat.json
    echo """set -vex -o pipefail
fextract2stat ${input_fextract_csv} ${output_stat_json}
""" > $out_dir/chunk-${chunk_id}.genstat.sh
    $pitch7/bin/qu 4 "bash $out_dir/chunk-${chunk_id}.genstat.sh"
}

function gen_npz() {
    chunk_id=$1
    x=$2
    y=$3
    z=$4
    k=$5
    num_train_rows=$6

    echo "chunk_id: ${chunk_id}"
    echo "X:Y:Z=$x,$y,$z"
    echo "K=$k"
    out_dir=${fextract_dir}/XYZK_${x}_${y}_${z}_${k}
    input_fextract_csv=${out_dir}/chunk-${chunk_id}.fextract.csv
    output_stat_json=${out_dir}/chunk-${chunk_id}.stat.json
    output_prefix=${out_dir}/chunk-${chunk_id}.nrows.${num_train_rows}

    echo "Input fextract csv: ${input_fextract_csv}"
    echo "Output stat json: ${output_stat_json}"
    echo "Number of training rows: ${num_train_rows}"

    echo "Convert fextract.csv to npz, standardize features and extract Y arrays"
    echo """set -vex -o pipefail
    fextract2numpy ${input_fextract_csv} ${output_prefix} --stat-json ${output_stat_json} --num-train-rows ${num_train_rows}
""" > $out_dir/chunk-${chunk_id}.gennpz.sh
    $pitch7/bin/qu 8 "bash $out_dir/chunk-${chunk_id}.gennpz.sh"
}


function multinomial_naive() {
    chunk_id=$1
    x=$2
    y=$3
    z=$4
    k=$5
    num_train_rows=$6

    echo "Chunk ID: ${chunk_id}"
    echo "X:Y:Z=$x,$y,$z"
    echo "K=$k"
    out_dir=${fextract_dir}/XYZK_${x}_${y}_${z}_${k}
    input_stat_json=${out_dir}/chunk-${chunk_id}.stat.json
    input_train_npz=${out_dir}/chunk-${chunk_id}.nrows.${num_train_rows}.train.npz
    model_dir=${out_dir}/models/chunk-${chunk_id}.nrows.${num_train_rows}/multinom.naive
    model_id=0
    mkdir -p ${model_dir}
    echo """set -vex -o pipefail

echo "Train on ${input_train_npz}"
cp ${out_dir}/chunk-${chunk_id}.stat.json ${model_dir}/features.stat.json
cp ${out_dir}/chunk-${chunk_id}.nrows.${num_train_rows}.features.order.json ${model_dir}/features.orders.json
cp ${out_dir}/chunk-${chunk_id}.nrows.1000000.base_map_probability.json ${model_dir}/base_map_probability.json
multinomial ${input_train_npz} ${model_dir} --name 'lambda-multinomial-naive' --batch-size 512 --epochs 1000 --model-id ${model_id}

echo "Check saved model"
saved_model_cli show --dir ${model_dir} --tag_set serve --signature_def serving_default
""" > ${model_dir}/multinomial_naive.sh

    bash ${model_dir}/multinomial_naive.sh
}

function patch_a_model() {
    chunk_id=$1
    x=$2
    y=$3
    z=$4
    k=$5
    num_train_rows=$6

    echo "Chunk ID: ${chunk_id}"
    echo "X:Y:Z=$x,$y,$z"
    echo "K=$k"

    . /mnt/software/Modules/current/init/bash 
    module load ccsqv/develop R
    out_dir=${fextract_dir}/XYZK_${x}_${y}_${z}_${k}
    input_stat_json=${out_dir}/chunk-${chunk_id}.stat.json
    input_train_npz=${out_dir}/chunk-${chunk_id}.nrows.${num_train_rows}.train.npz
    model_dir=${out_dir}/models/chunk-${chunk_id}.nrows.${num_train_rows}/multinom.naive

    mkdir -p ${model_dir}
    cp ${out_dir}/chunk-${chunk_id}.stat.json ${model_dir}/features.stat.json
    cp ${out_dir}/chunk-${chunk_id}.nrows.${num_train_rows}.features.order.json ${model_dir}/features.order.json

    baseqv_csv=${model_dir}/chunk-${chunk_id}.baseqv.csv
    sampling_json=${out_dir}/chunk-${chunk_id}.nrows.${num_train_rows}.base_map_probability.json
    pop_json=${model_dir}/chunk-${chunk_id}.population.base_map_probabilty.json
    qvtools ${original_db_dir}/ArrowQv/chunk-${chunk_id}.ccs2genome.bam ${baseqv_csv} -j 16
    Rscript --vanilla /mnt/software/c/ccsqv/master/bin/R/base_map_prob_json_from_baseqv_csv.R -i ${baseqv_csv} -o  ${pop_json}
    merge-base-map-prob ${sampling_json} ${pop_json} ${model_dir}/base_map_probability.json
    rm -f out.bam
    applyqv \
        $PI/ccsqv/tests/data/applyqv/one-read.sr2ccs.bam \
        $PI/ccsqv/tests/data/applyqv/one-read.ccs2genome.bam out.bam \
        -p ${model_dir} \
        --log-level INFO
    samtools view out.bam |cut -f 1-4
}

function eval_a_model() {
    chunk_id=$1
    x=$2
    y=$3
    z=$4
    k=$5
    num_train_rows=$6
    test_chunk_id=$7

    echo "Chunk ID: ${chunk_id}"
    echo "X:Y:Z=$x,$y,$z"
    echo "K=$k"
    echo "Test chunk ID: ${test_chunk_id}"
    out_dir=${fextract_dir}/XYZK_${x}_${y}_${z}_${k}
    input_stat_json=${out_dir}/chunk-${chunk_id}.stat.json
    test_fextract_csv=${out_dir}/chunk-${test_chunk_id}.fextract.csv
    model_dir=${out_dir}/models/chunk-${chunk_id}.nrows.${num_train_rows}/multinom.naive
    tmp_dir=$(mktemp -d -t evl-multinom-naive-XXXXXXXXXX -p /tmp/yli)
    echo """set -vex -o pipefail
ls ${test_fextract_csv}
mkdir -p ${tmp_dir}
rm -f ${model_dir}/test.original.fextract.csv
ln -s ${test_fextract_csv} ${model_dir}/test.original.fextract.csv
fextract2numpy ${test_fextract_csv} ${tmp_dir}/test --stat-json ${input_stat_json} --num-train-rows ${num_train_rows}
evalmodel ${model_dir} ${tmp_dir}/test.train.npz ${tmp_dir}/test.train.fextract.csv  ${model_dir}/test.eval.csv | tail -n 2 > ${model_dir}/test.eval.log
""" > ${model_dir}/eval.sh
    bash ${model_dir}/eval.sh
}

function apply_a_model(){
    . /mnt/software/Modules/current/init/bash 
    module load ccsqv/develop
    chunk_id=$1
    x=$2
    y=$3
    z=$4
    k=$5
    num_train_rows=$6

    echo "Apply the input model to 1% of lambda digest movies"

    echo "Chunk ID: ${chunk_id}"
    echo "X:Y:Z=$x,$y,$z"
    echo "K=$k"
    out_dir=${fextract_dir}/XYZK_${x}_${y}_${z}_${k}

    input=/pbi/dept/consensus/ccsqv/data/Mule/lambda/one_percent.lambda.arrowqv.ccs2genome.tsv
    model_dir=${out_dir}/models/chunk-${chunk_id}.nrows.${num_train_rows}/multinom.naive
    name=XYZK_${x}_${y}_${z}_${k}_multinom_naive
    rm -rf ${model_dir}/wdl
    #$pitch7/bin/qu 8 "cromwell-ccsqv-apply.sh ${input} ${model_dir} ${name} ${model_dir}/wdl"
    cromwell-ccsqv-apply.sh ${input} ${model_dir} ${name} ${model_dir}/wdl
}



function batch_gen_npz() {
nrows=1000000
for chunk_id in `echo 1 9 70`; do \
    for k in `echo 20 30`; do \
        gen_npz ${chunk_id} 40 40 20 $k $nrows; \
        gen_npz ${chunk_id} 30 30 40 $k $nrows; \
        gen_npz ${chunk_id} 20 20 60 $k $nrows; \
        gen_npz ${chunk_id} 10 10 80 $k $nrows; \
        gen_npz ${chunk_id} 5 5 90 $k $nrows; \
    done
done
}

function batch_trian_multinom_naive() {
test_chunk_id=70
for chunk_id in `echo 1 9 70`; do \
    for k in `echo 20 30`; do \
        multinomial_naive ${chunk_id} 40 40 20 $k $nrows; \
        multinomial_naive ${chunk_id} 30 30 40 $k $nrows; \
        multinomial_naive ${chunk_id} 20 20 60 $k $nrows; \
        multinomial_naive ${chunk_id} 10 10 80 $k $nrows; \
        multinomial_naive ${chunk_id} 5 5 90 $k $nrows ; \
    done
done
}


function batch_eval_models() {
test_chunk_id=70
for chunk_id in `echo 1 9 70`; do \
    for k in `echo 20 30`; do \
        eval_a_model ${chunk_id} 40 40 20 $k $nrows ${test_chunk_id}; \
        eval_a_model ${chunk_id} 30 30 40 $k $nrows ${test_chunk_id}; \
        eval_a_model ${chunk_id} 20 20 60 $k $nrows ${test_chunk_id}; \
        eval_a_model ${chunk_id} 10 10 80 $k $nrows ${test_chunk_id}; \
        eval_a_model ${chunk_id} 5 5 90 $k $nrows ${test_chunk_id}; \
    done
done
}

function pipeline() {
    chunk_id=$1
    x=$2
    y=$3
    z=$4
    k=$5
    num_train_rows=$6

    # sampling_xyz ${chunk_id} $x $y $z $k
    # gen_stat ${chunk_id} $x $y $z $k
    # gen_npz ${chunk_id}  $x $y $z $k $num_train_rows
    # multinomial_naive ${chunk_id} $x $y $z $k $num_train_rows
    # patch_a_model ${chunk_id} $x $y $z $k $num_train_rows
    apply_a_model ${chunk_id} $x $y $z $k $num_train_rows
}

chunk_id=9
x=1
y=1
z=98
k=20
num_train_rows=1000000
pipeline ${chunk_id} $x $y $z $k ${num_train_rows}

# batch_trian_multinom_naive
# batch_eval_models
