python main.py  --epochs 1 --train-batch-size 64 --num-samples 2000 \
               --bits '16,32,48,64' --exp-name 'asy' --machine-name 1080 \
               --topk '54000' --dataset-name 'cifar10' \
               --test-interval 10 --max-iter 40 \
               --learning-rate 0.001