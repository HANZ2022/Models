建议把小图和原图的文件名改成一一对应的.

示例指令(一般就是修改对应的输入/输出/HR文件夹路径，权重路径（若有），scale factor就行)：
bicubic:
python Bicubic.py --input_folder "./DIV2K/DIV2K_valid_LR_bicubic/X2" --output_folder "./Output/DIV2K_X2/bicubic/results/SR2" 2 --overwrite --image_extensions png

FSRCNN:
cd FSRCNN-PyTorch-1
python validate.py
参数在config.py调

IMDN: 此算法默认把测试和验证数据集放在./IMDN/Test_Datasets里
cd IMDN
python test_IMDN.py --test_hr_folder Test_Datasets/DIV2K_valid_HR/ --test_lr_folder Test_Datasets/DIV2K_valid_LR_bicubic/X2/ --output_folder ../Output/DIV2K_X2/IMDN/results/SR2/ --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 2

RT4KSR:此算法默认把测试和验证数据集放在./dataset/testsets/xxx（数据集名字）/val里。is_train作者要求保留。只有*2/*3
./code/data/benchmark.py里是支持的数据集（只是一个名字，如果需要跑其他数据集的话照着修改就行）
cd RT4KSR
python code/test.py --dataroot ./dataset --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --benchmark div2k --is-train

psnr和ssim：submission_id用对应算法的名字（比如FSRCNN），原脚本用了一个有点麻烦的规则来定位输出SR文件夹的位置
python ./NTIRE23-RTSR/demo/calc_metrics.py --submission-id FSRCNN --sr-dir ./Output/DIV2K_X4 --gt-dir ./DIV2K/DIV2K_valid_HR

lpips：
python ./LPIPS/lpips_2dirs.py -d0 ./DIV2K/DIV2K_valid_HR -d1 ./Output/DIV2K_X4/bicubic/results/SR2 -o results_lpips.txt --use_gpu
