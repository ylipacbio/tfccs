set -vex -o pipefail
# This documented how I preprocess lambda digest dataset and train
# a multinomial model regression to CCS to genome mapping.
movie=m64002_190608_021007
echo "Movie: $movie"
original_db_dir=/pbi/dept/consensus/ccsqv/data/Mule/lambda/bam/m64002_190608_021007/V2Bmark_SMS_Beta2_LambdaDigestEagI_SP2p1_DA011306_FCR_pkmid500_3260307/
fextract_dir=/pbi/dept/secondary/siv/yli/jira/tak-230-tf-sampling

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


chunk_id=9
k=20
nrows=1000000
# multinomial_naive ${chunk_id} 40 40 20 $k $nrows
test_chunk_id=70
#eval_a_model ${chunk_id} 40 40 20 $k $nrows ${test_chunk_id}

#batch_trian_multinom_naive
batch_eval_models
