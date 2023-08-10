for B in 16 64 256 1024; do
  for E in 1 5 10; do
    for C in 0.0 0.2 0.4 0.6 0.8 1.0; do
      python -u main.py \
        --batch_size $B \
        --local_epoch $E \
        --fraction $C
    done
  done
done
