# SwinUnetr
## train (--loss={CE, Dice, DiceCE})
python main.py --logdir=benchmark_brats21_nested --save_checkpoint --use_checkpoint --noamp --val_every=50 --max_epochs=200 --loss=Dice --fold=0
## do_TS
python do_TS.py --pretrained_dir=benchmark_brats21_nested --loss=Dice --fold=0 --TS=list_1000
## predict npz for ECE
python test.py --pretrained_dir=benchmark_brats21_nested --loss=Dice --fold=0 --ECE
python test.py --pretrained_dir=benchmark_brats21_nested --loss=Dice --fold=0 --ECE --TS list_1000
## predict npz for metrics
python test.py --pretrained_dir=benchmark_brats21_nested --loss=Dice --fold=0
python test.py --pretrained_dir=benchmark_brats21_nested --loss=Dice --fold=0 --TS list_1000


# ours
#--i=benchmark_brats21_nested --fold=0
load from prob.npz from 'benchmark_brats21_nested/fold_0/results'
# if args.TS is not None: _{args.TS}
load gt.nii.gz from f'/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw/Brats2021_Training_Data/BraTS2021_01146/{basename}_seg.nii.gz'
# val
python ratio_estimator.py  --biomarker ntr --ce_type bins15 /staging/leuven/stg_00081/jli/calibration/monai_research/SwinUNETR/BRATS21/benchmark_brats21_nested/fold_0/validation_ece  /staging/leuven/stg_00081/jli/calibration/monai_research/SwinUNETR/BRATS21/benchmark_brats21_nested/fold_0/validation_ece
# test
python ratio_estimator.py  --biomarker ntr --ce_type bins15 /staging/leuven/stg_00081/jli/calibration/monai_research/SwinUNETR/BRATS21/benchmark_brats21_nested/fold_0/test  /staging/leuven/stg_00081/jli/calibration/monai_research/SwinUNETR/BRATS21/benchmark_brats21_nested/fold_0/test


