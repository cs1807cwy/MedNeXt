
# Set Envs
export nnUNet_raw_data_base="/data/cwy/datasets/MedNeXt/nnUNet_raw_data_base"
export nnUNet_preprocessed="/data/cwy/datasets/MedNeXt/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/cwy/datasets/MedNeXt/nnUNet_trained_models"

# Task settings
task_id=301
dataset="Task${task_id}_Esophagus_mix_1_2"

# plan and preprocess
mednextv1_plan_and_preprocess -t ${task_id} -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d None
log_dir="/data/cwy/datasets/MedNeXt/logs"
export CUDA_VISIBLE_DEVICES=4 && nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 ${task_id} 0 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_0.log" &
export CUDA_VISIBLE_DEVICES=5 && nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 ${task_id} 1 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_1.log" &
export CUDA_VISIBLE_DEVICES=6 && nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 ${task_id} 2 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_2.log" &
export CUDA_VISIBLE_DEVICES=7 && nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 ${task_id} 3 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_3.log" &
export CUDA_VISIBLE_DEVICES=3 && nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 ${task_id} 4 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_4.log" &

# find best
mednextv1_find_best_configuration -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -pl nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres

# predict with model_best
add_cmd="--save_npz"
input="${nnUNet_raw_data_base}/nnUNet_raw_data/${dataset}/imagesTs"
output="${RESULTS_FOLDER}/nnUNet/3d_fullres/${dataset}/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1"
export CUDA_VISIBLE_DEVICES=0 && mednextv1_predict -i ${input} -o ${output}/fold_0/test_model_best -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 0 -chk model_best ${add_cmd}
export CUDA_VISIBLE_DEVICES=1 && mednextv1_predict -i ${input} -o ${output}/fold_1/test_model_best -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 1 -chk model_best ${add_cmd}
export CUDA_VISIBLE_DEVICES=2 && mednextv1_predict -i ${input} -o ${output}/fold_2/test_model_best -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 2 -chk model_best ${add_cmd}
export CUDA_VISIBLE_DEVICES=3 && mednextv1_predict -i ${input} -o ${output}/fold_3/test_model_best -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 3 -chk model_best ${add_cmd}
export CUDA_VISIBLE_DEVICES=4 && mednextv1_predict -i ${input} -o ${output}/fold_4/test_model_best -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 4 -chk model_best ${add_cmd}
mednextv1_ensemble -f ${output}/fold_0/test_model_best ${output}/fold_1/test_model_best ${output}/fold_2/test_model_best ${output}/fold_3/test_model_best ${output}/fold_4/test_model_best -o ${output}/ensemble/test_model_best -pp ${output}/fold_0/test_model_best/postprocessing.json

# predict with model_final
add_cmd="--save_npz"
input="${nnUNet_raw_data_base}/nnUNet_raw_data/${dataset}/imagesTs"
output="${RESULTS_FOLDER}/nnUNet/3d_fullres/${dataset}/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1"
export CUDA_VISIBLE_DEVICES=0 && mednextv1_predict -i ${input} -o ${output}/fold_0/test_model_final -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 0 -chk model_final_checkpoint ${add_cmd}
export CUDA_VISIBLE_DEVICES=1 && mednextv1_predict -i ${input} -o ${output}/fold_1/test_model_final -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 1 -chk model_final_checkpoint ${add_cmd}
export CUDA_VISIBLE_DEVICES=2 && mednextv1_predict -i ${input} -o ${output}/fold_2/test_model_final -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 2 -chk model_final_checkpoint ${add_cmd}
export CUDA_VISIBLE_DEVICES=3 && mednextv1_predict -i ${input} -o ${output}/fold_3/test_model_final -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 3 -chk model_final_checkpoint ${add_cmd}
export CUDA_VISIBLE_DEVICES=4 && mednextv1_predict -i ${input} -o ${output}/fold_4/test_model_final -t ${task_id} -tr nnUNetTrainerV2_MedNeXt_M_kernel3 -p nnUNetPlansv2.1_trgSp_1x1x1 -m 3d_fullres -f 4 -chk model_final_checkpoint ${add_cmd}
mednextv1_ensemble -f ${output}/fold_0/test_model_final ${output}/fold_1/test_model_final ${output}/fold_2/test_model_final ${output}/fold_3/test_model_final ${output}/fold_4/test_model_final -o ${output}/ensemble/test_model_final -pp ${output}/fold_0/test_model_final/postprocessing.json

# evaluate metrics
pred_dir="${RESULTS_FOLDER}/nnUNet/3d_fullres/${dataset}/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1"
gt_dir="${nnUNet_raw_data_base}/nnUNet_raw_data/${dataset}/labelsTs"
test_dir=test_model_best
#test_dir=test_model_final
export fold=0 && mednextv1_evaluate_folder -l 1 -ref ${gt_dir} -pred ${pred_dir}/fold_${fold}/${test_dir}
export fold=1 && mednextv1_evaluate_folder -l 1 -ref ${gt_dir} -pred ${pred_dir}/fold_${fold}/${test_dir}
export fold=2 && mednextv1_evaluate_folder -l 1 -ref ${gt_dir} -pred ${pred_dir}/fold_${fold}/${test_dir}
export fold=3 && mednextv1_evaluate_folder -l 1 -ref ${gt_dir} -pred ${pred_dir}/fold_${fold}/${test_dir}
export fold=4 && mednextv1_evaluate_folder -l 1 -ref ${gt_dir} -pred ${pred_dir}/fold_${fold}/${test_dir}

# evaluate ensembled metrics
gt_dir="${nnUNet_raw_data_base}/nnUNet_raw_data/${dataset}/labelsTs"
test_dir="test_model_best"
#test_dir="test_model_final"
pred_dir="${RESULTS_FOLDER}/nnUNet/3d_fullres/${dataset}/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1"
mednextv1_evaluate_folder -l 1 -ref ${gt_dir} -pred ${pred_dir}/ensemble/${test_dir}
