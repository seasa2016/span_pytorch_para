# An Empirical Study of Span Representation in Argumentation Structure Parsing
## Citation
```
@InProceedings{P19-1464,
  author = "Kuribayashi, Tatsuki
	and Ouchi, Hiroki
	and Inoue, Naoya
	and Reisert, Paul
    	and Miyoshi, Toshinori
    	and Suzuki, Jun
    	and Inui, Kentaro"
  title = 	"An Empirical Study of Span Representation in Argumentation Structure Parsing",
  booktitle = 	"Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
  year = 	"2019",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"4691-4698",
  location = 	"Florence, Italy",
  url = 	"https://www.aclweb.org/anthology/P19-1464"
}
```
- conference paper: https://www.aclweb.org/anthology/P19-1464

## Prerequirement
- python=3.6
- pytorch
- allennlp  

## preprocess
### please do the bio parsing first
#### preprocess from the bio parsing output
python preprocess_tree.py "predict file" "mapping file" "base folder"
#### predict the elmo embedding
allennlp elmo work/PE4ELMo.tsv work/PE4ELMo.hdf5 --all --cuda-device $1
allennlp elmo work/MT4ELMo.tsv work/MT4ELMo.hdf5 --all --cuda-device $1


## Trainging
python train.py --use-elmo 1 --data-path ${PATH_TO_DATA} --elmo-path ${PATH_TO_ELMO_EMBEDDING} --optimizer Adam --lr 0.003 --ac-type-alpha 0.25 --link-type-alpha 0.25 --batchsize 16 --epoch 64 --dropout 0.5 --dropout-lstm 0.1 --lstm-ac --lstm-shell --lstm-ac-shell --lstm-type --elmo-layers avg --train --dev --test --save-dir ./saved_model/

## Testing
python test.py --test-data ${PATH_TO_DATA} --test-elmo ${PATH_TO_ELMO_EMBEDDING} --result-path ${OUTPUT_PATH} --save-model ${CHECK_POINT} --lstm-ac --lstm-shell --lstm-ac-shell --lstm-type --use-elmo 1  --elmo-layers avg

