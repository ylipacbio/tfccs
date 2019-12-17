set -vex -o pipefail
echo Add features.order.json, base_map_probability.json, and features.stat.json to TFmodels and patch base_map_probability.json,

original_db_dir=/pbi/dept/consensus/ccsqv/data/Mule/lambda/bam/m64002_190608_021007/V2Bmark_SMS_Beta2_LambdaDigestEagI_SP2p1_DA011306_FCR_pkmid500_3260307/
fextract_dir=/pbi/dept/secondary/siv/yli/jira/tak-230-tf-sampling

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
    /mnt/software/c/ccsqv/master/bin/qvtools ${original_db_dir}/ArrowQv/chunk-${chunk_id}.ccs2genome.bam ${baseqv_csv} -j 16
    Rscript --vanilla /mnt/software/c/ccsqv/master/bin/R/base_map_prob_json_from_baseqv_csv.R -i ${baseqv_csv} -o  ${pop_json}
    merge-base-map-prob ${sampling_json} ${pop_json} ${model_dir}/base_map_probability.json
    rm -f out.bam
    /mnt/software/c/ccsqv/master/bin/applyqv \
        $PI/ccsqv/tests/data/applyqv/one-read.sr2ccs.bam \
        $PI/ccsqv/tests/data/applyqv/one-read.ccs2genome.bam out.bam \
        -p ${model_dir} \
        --log-level INFO
    samtools view out.bam |cut -f 1-4
}

function batch_patch_model() {
num_train_rows=1000000
for chunk_id in `echo 1 9 70`; do \
    for k in `echo 20 30`; do \
        patch_a_model ${chunk_id} 40 40 20 $k $num_train_rows; \
        patch_a_model ${chunk_id} 30 30 40 $k $num_train_rows; \
        patch_a_model ${chunk_id} 20 20 60 $k $num_train_rows; \
        patch_a_model ${chunk_id} 10 10 80 $k $num_train_rows; \
        patch_a_model ${chunk_id} 5 5 90 $k $num_train_rows; \
    done
done
}

chunk_id=1
x=20
y=20
z=60
k=20
num_train_rows=1000000
# patch_a_model ${chunk_id} $x $y $z $k ${num_train_rows}

batch_patch_model
