Install from git
```sh
git clone https://github.com/deepmipt/deeppavlov
cd deeppavlov
pip3 install Cython==0.26.0
pip3 install -r requirements.txt
python3 setup.py develop
```

Run model training
```sh
pyb train_<model1_name>[ ,train_<model2_name>, ...]
```
Run model testing
```sh
pyb run_unit_tests -P unittest_test_method_prefix="test_<model1_name>[,test_<model2_name>,...]"
```
where <modelX_name> should be replaced with the name of the model from the library: ner, paraphraser, insults, squad, coreference, coref.

To test all the models use
```sh
pyb run_unit_tests -v
```
After that, test metrics could be found within ./build/kpi_score_reports.txt file and detailed output could be found in target/reports.

Works on Ubuntu 16.04.

Install from internal iPavlov PyPi server as dependency for your project
```sh
pip install --extra-index-url http://{host of internal iPavlov PyPi server}:{port}/ --trusted-host {host of internal iPavlov PyPi server} deeppavlov
```
