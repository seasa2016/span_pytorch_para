python test.py --test-data ${1}/tree/pre_neg.jsons --test-elmo ${1}/tree/cmv_elmo_neg.hdf5 --result-path ${2}/pred_neg --save-model ./new_data_1/qq34/checkpoint_11.pt --lstm-ac --lstm-shell --lstm-ac-shell --lstm-type --use-elmo 1  --elmo-layers avg
python test.py --test-data ${1}/tree/pre_pos.jsons --test-elmo ${1}/tree/cmv_elmo_pos.hdf5 --result-path ${2}/pred_pos --save-model ./new_data_1/qq34/checkpoint_11.pt --lstm-ac --lstm-shell --lstm-ac-shell --lstm-type --use-elmo 1  --elmo-layers avg

