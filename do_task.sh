
# Set Envs
export nnUNet_raw_data_base="/data/cwy/datasets/MedNeXt/nnUNet_raw_data_base"
export nnUNet_preprocessed="/data/cwy/datasets/MedNeXt/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/cwy/datasets/MedNeXt/nnUNet_trained_models"

# Task settings
task_id=200
dataset="Task${task_id}_Synapse_Abdomen"
model="2d"

# plan and preprocess
#mednextv1_plan_and_preprocess -t ${task_id} -pl3d None -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1
log_dir="/data/cwy/datasets/MedNeXt/logs"
export CUDA_VISIBLE_DEVICES=5 && nohup mednextv1_train ${model} nnUNetTrainerV2_MedNeXt_M_kernel3_2d ${task_id} 0 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_0.log" &
#export CUDA_VISIBLE_DEVICES=5 && nohup mednextv1_train ${model} nnUNetTrainerV2_MedNeXt_M_kernel3_2d ${task_id} 1 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_1.log" &
#export CUDA_VISIBLE_DEVICES=6 && nohup mednextv1_train ${model} nnUNetTrainerV2_MedNeXt_M_kernel3_2d ${task_id} 2 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_2.log" &
#export CUDA_VISIBLE_DEVICES=7 && nohup mednextv1_train ${model} nnUNetTrainerV2_MedNeXt_M_kernel3_2d ${task_id} 3 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_3.log" &
#export CUDA_VISIBLE_DEVICES=3 && nohup mednextv1_train ${model} nnUNetTrainerV2_MedNeXt_M_kernel3_2d ${task_id} 4 -p nnUNetPlansv2.1_trgSp_1x1x1 > "${log_dir}/${dataset}_crossval_results_folds_4.log" &
