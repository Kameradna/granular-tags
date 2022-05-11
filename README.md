# granular-tags
Granular tag prediction for IU-x-ray

Just run
>bash run.sh
for downloading and extracting IU-xray data
downloading big-transfer models
run training runs and generate logs


<!-- labelnosum should have length 369 and has length 353 -->
xor_for_hamming should have length 369 and has length 369
Traceback (most recent call last):
  File "/home/prowe/anaconda3/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/prowe/anaconda3/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/prowe/granular-tags/big_transfer/bit_pytorch/train.py", line 442, in <module>
    main(parser.parse_args())
  File "/home/prowe/granular-tags/big_transfer/bit_pytorch/train.py", line 427, in main
    run_eval(model, valid_loader, device, chrono, logger, args, step='end')
  File "/home/prowe/granular-tags/big_transfer/bit_pytorch/train.py", line 245, in run_eval
    label_density = np.mean(labelnosum/len(tp,1))
TypeError: len() takes exactly one argument (2 given)
